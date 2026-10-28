"""Load the pinned TI 2D detector without importing unrelated pose/data tools.

Architecture flags match exps/default/yolox_nano_ti_lite.py at
cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6. Vendor source is unchanged.
"""
import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party/edgeai-yolox/yolox'


def load_vendor():
    # Importing the historical package normally eagerly imports 6D-pose tools
    # and CUDA-oriented training extensions. Only the official 2D modules below
    # are needed for the CPU loop; no layer/loss implementation is replaced.
    if 'yolox.models.yolo_head' in sys.modules:
        return
    for name, path in [('yolox',VENDOR),('yolox.utils',VENDOR/'utils'),('yolox.models',VENDOR/'models')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    boxes = importlib.import_module('yolox.utils.boxes')
    sys.modules['yolox.utils'].bboxes_iou = boxes.bboxes_iou


class FlameDetector(nn.Module):
    def __init__(self):
        super().__init__()
        load_vendor()
        backbone = importlib.import_module('yolox.models.yolo_pafpn').YOLOPAFPN
        head = importlib.import_module('yolox.models.yolo_head').YOLOXHead
        self.backbone = backbone(.33,.25,in_channels=[256,512,1024],act='relu',depthwise=False,
                                 conv_focus=True,split_max_pool_kernel=True)
        self.head = head(1,.25,in_channels=[256,512,1024],act='relu',depthwise=False)
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3
                layer.momentum = .03
        self.head.initialize_biases(1e-2)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        if self.training:
            losses = self.head(features, targets, images)
            return losses[0]
        return self.head(features)


def load_pretrained(model, path):
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    state = checkpoint.get('model',checkpoint.get('state_dict',checkpoint))
    expected = model.state_dict()
    matched = {k:v for k,v in state.items() if k in expected and v.shape == expected[k].shape}
    incompatible = model.load_state_dict(matched,strict=False)
    missing = incompatible.missing_keys
    # The only expected mismatch is the newly initialized one-class prediction.
    unexpected = [k for k in missing if not k.startswith('head.cls_preds.')]
    if unexpected:
        raise RuntimeError(f'Pretrained architecture mismatch: {unexpected}')
    return {'loaded_tensors':len(matched),'new_class_tensors':missing}
