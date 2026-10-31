"""Evaluate the selected EMA checkpoint and export a checked ONNX artifact."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import Flames, sha256
from metrics import evaluate
from model import Detector, TI_COMMIT
from train import atomic_json


def export(model, dataset, output):
    import onnx
    import onnxruntime as ort
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    x = dataset[0][0].unsqueeze(0)
    path = output / 'flame.onnx'
    torch.onnx.export(model, x, str(path), opset_version=17,
                      input_names=['images'], output_names=['detections'],
                      do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    positive = next((i for i, r in enumerate(dataset.records) if r['boxes']), 0)
    negative = next((i for i, r in enumerate(dataset.records) if not r['boxes']), 0)
    indices = list(dict.fromkeys([positive, negative, 0, len(dataset)-1]))
    box_errors, score_errors = [], []
    with torch.inference_mode():
        for index in indices:
            sample = dataset[index][0].unsqueeze(0)
            expected = model(sample).numpy()
            actual = session.run(None, {'images': sample.numpy()})[0]
            box_errors.append(float(np.abs(expected[:, :, :4]-actual[:, :, :4]).max()))
            score_errors.append(float(np.abs(expected[:, :, 4:]-actual[:, :, 4:]).max()))
            # Boxes are input PIXELS, scores are dimensionless. A shared relative
            # tolerance is inappropriate near box coordinate zero. Bound boxes
            # absolutely to .05 input pixel (less than 1/10 pixel) and scores separately.
            np.testing.assert_allclose(actual[:, :, :4], expected[:, :, :4], atol=.05, rtol=0)
            np.testing.assert_allclose(actual[:, :, 4:], expected[:, :, 4:], atol=1e-5, rtol=.001)
    contract = dict(architecture='TI YOLOv9-S-Lite one-class adapter', ti_commit=TI_COMMIT,
                    input=dict(name='images', shape=list(x.shape), dtype='float32', channels='RGB',
                               values='0..1', resize='aspect preserving, rounded dimensions',
                               padding='top-left content, bottom/right raw pad 114',
                               box_restore='divide x/y by actual rounded resize scale_x/scale_y'),
                    output=dict(name='detections', shape=list(expected.shape),
                                fields=['x1', 'y1', 'x2', 'y2', 'flame_probability'],
                                coordinates='input pixels; raw pre-NMS', confidence='class sigmoid; no objectness multiplication'),
                    postprocessing=dict(score_floor=.001, nms_iou=.65, max_detections=100),
                    board_compilation='pending; ONNX validity is not target execution proof')
    atomic_json(contract, output/'model_contract.json')
    result = dict(onnx_checker=True, parity_passed=True, parity_inputs=len(indices),
                  box_max_absolute_error_pixels=max(box_errors), box_atol_pixels=.05, box_rtol=0,
                  score_max_absolute_error=max(score_errors), score_atol=1e-5, score_rtol=.001,
                  sha256=sha256(path), bytes=path.stat().st_size,
                  parameters_inference=sum(p.numel() for module in (model.backbone, model.neck, model.bbox_head)
                                           for p in module.parameters()))
    atomic_json(result, output/'export.json')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensorlab', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['validation_full', 'test'], default='validation_full')
    p.add_argument('--final-test', action='store_true')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--skip-export', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.split == 'test' and not a.final_test:
        p.error('Test requires --final-test; tune/checkpoint selection must use validation')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    for name, expected in checkpoint['identity']['manifests'].items():
        if sha256(Path(a.manifests)/f'{name}.json') != expected:
            raise RuntimeError('Training/monitor manifests changed since model selection')
    size = checkpoint['identity']['config']['size']
    manifest = Path(a.manifests)/f'{a.split}.json'
    dataset = Flames(json.loads(manifest.read_text()), size)
    model = Detector(a.tensorlab)
    model.load_state_dict(checkpoint['model'], strict=True)
    device = 'cpu' if a.cpu else 'cuda'
    model.to(device)
    metrics, predictions = evaluate(model, dataset, a.batch, a.workers)
    metrics.update(split=a.split, checkpoint_sha256=sha256(a.checkpoint),
                   checkpoint_epoch=checkpoint['epoch'], manifest_sha256=sha256(manifest),
                   evaluation_device=str(next(model.parameters()).device),
                   gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                   caveats=['Kaggle test has already been examined in prior experiments; independence is not established',
                            'Cross-split duplicate policy is recorded in prepare_report.json',
                            'Different architectures/resolutions/recipes prevent attribution to architecture alone'])
    atomic_json(metrics, output/'metrics.json')
    atomic_json(predictions, output/'predictions.json')
    if not a.skip_export:
        export(model, dataset, output/'export')
    c, op = metrics['coco'], metrics['at_confidence_0.40']
    report = '\n'.join([
        '# YOLOv9-S-TI-Lite measured results', '',
        f"Split: {a.split}; images: {len(dataset)}; selected epoch: {checkpoint['epoch']}.", '',
        '| Metric | Measured value |', '|---|---:|',
        *[f'| {key} | {value:.6f} |' if value is not None else f'| {key} | unavailable |' for key, value in c.items()],
        *[f'| {key} (confidence .40, IoU .50) | {op[key]} |' for key in ('precision', 'recall', 'F1', 'TP', 'FP', 'FN')],
        '', 'AP is COCO bbox AP; FP counts boxes, not video alarm events.',
        'AM62A speed, memory, quantized accuracy, and 1 TOPS compliance have not been measured.',
        'See metrics.json for checkpoint/manifest hashes and exact definitions.', ''])
    (output/'REPORT.md').write_text(report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
