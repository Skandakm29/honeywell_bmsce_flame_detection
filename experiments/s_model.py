"""YOLOX-S-ti-lite with a one-class flame head, built from the same pinned TI modules as the baseline.

Flags match exps/default/yolox_s_ti_lite.py at cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6:
depth 0.33, width 0.50, ReLU, convolutional stem, split max-pooling, regular convolutions.
Only the width differs from the baseline FlameDetector (0.25), so forward/loss are inherited.
"""
import importlib
import sys
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from flame_model import FlameDetector, load_vendor  # noqa: E402

ARCHITECTURE = 'TI YOLOX-S-ti-lite'


class SFlameDetector(FlameDetector):
    def __init__(self):
        nn.Module.__init__(self)
        load_vendor()
        backbone = importlib.import_module('yolox.models.yolo_pafpn').YOLOPAFPN
        head = importlib.import_module('yolox.models.yolo_head').YOLOXHead
        self.backbone = backbone(.33,.5,in_channels=[256,512,1024],act='relu',depthwise=False,
                                 conv_focus=True,split_max_pool_kernel=True)
        self.head = head(1,.5,in_channels=[256,512,1024],act='relu',depthwise=False)
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3
                layer.momentum = .03
        self.head.initialize_biases(1e-2)


def build(arch):
    """'nano' is the unchanged baseline model; 's' is this experiment's model."""
    return {'nano':FlameDetector,'s':SFlameDetector}[arch]()
