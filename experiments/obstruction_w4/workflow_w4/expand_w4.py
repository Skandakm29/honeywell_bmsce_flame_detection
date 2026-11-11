"""Grow the trained one-class flame model into flame + obstruction WITHOUT touching the flame weights.

In the TI YOLOv9-S-Lite head only the last 1x1 classifier conv of each of the 6 class branches (3 main + 3 aux)
depends on the number of classes: weight [C, 128, 1, 1], bias [C]. Everything else (backbone, neck, box branch,
the two 3x3 class convs) has the same shape and is copied as is.
  channel 0 (flame)       = the trained flame weights, copied exactly
  channel 1 (obstruction) = weight 0, bias logit(0.01)  (the standard 1% prior)
So before any training the expanded model gives EXACTLY the old flame scores, and 0.01 obstruction everywhere.
"""
import math

import torch

from data import sha256
from model import TI_COMMIT

PRIOR = .01
CLASSIFIER_SUFFIX = 'class_conv.2.'


def expand_state(weights, target_state):
    """Return a state dict shaped like `target_state`, filled from the 1-class `weights`."""
    out, expanded = {}, []
    for key, target in target_state.items():
        if key not in weights:
            raise RuntimeError(f'Checkpoint is missing {key}')
        source = weights[key]
        if source.shape == target.shape:
            out[key] = source.clone()
            continue
        if CLASSIFIER_SUFFIX not in key or source.shape[0] != 1 or source.shape[1:] != target.shape[1:]:
            raise RuntimeError(f'Unexpected shape change at {key}: {tuple(source.shape)} -> {tuple(target.shape)}')
        grown = torch.zeros_like(target)
        grown[:1] = source
        if key.endswith('bias'):
            grown[1:] = math.log(PRIOR / (1 - PRIOR))
        out[key] = grown
        expanded.append(key)
    extra = set(weights) - set(target_state)
    if extra:
        raise RuntimeError(f'Checkpoint has unexpected tensors: {sorted(extra)[:5]}')
    if len(expanded) != 12:
        raise RuntimeError(f'Expected 12 classifier tensors to grow (6 weights + 6 biases), got {len(expanded)}')
    return out, expanded


def load_expanded(model, checkpoint, expected_sha256, size):
    """Drop-in for train_v10.load_trained (same checks), for a 2-class model and a 1-class checkpoint."""
    if sha256(checkpoint) != expected_sha256:
        raise RuntimeError('Trained initialization checksum mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = state.get('identity')
    if identity is not None and (identity['ti_commit'] != TI_COMMIT or identity['config']['size'] != size):
        raise RuntimeError('Trained initialization architecture/input mismatch')
    weights = state['model']
    if not all(torch.isfinite(v).all() for v in weights.values() if v.is_floating_point()):
        raise FloatingPointError('Nonfinite trained initialization')
    grown, expanded = expand_state(weights, model.state_dict())
    model.load_state_dict(grown, strict=True)
    return dict(checkpoint_sha256=expected_sha256, source_epoch=state.get('epoch'), matched_tensors=len(grown),
                reinitialized=[], expanded=expanded, obstruction_prior=PRIOR,
                identity_checked=identity is not None, mode='trained_weights_expanded_to_2_classes')
