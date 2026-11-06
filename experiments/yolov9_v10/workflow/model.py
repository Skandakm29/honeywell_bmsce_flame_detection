"""One-class TI YOLOv9-S-Lite adapter; pinned vendor layers, no MMCV build.

Only six vendor modules are loaded under a private namespace. The auxiliary
head's unused MMCV ConvModule import is removed in memory. No vendor file is
modified. Graph surgery matches TI v1 for this architecture: SiLU -> ReLU and
large max pools -> sequential 3x3 pools. Pool size/type follow the TI Lite config.
"""
import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TI_COMMIT = 'f272548ef0fe96046f1bf2c3833e040a1ecf8e44'
PRETRAINED_SHA256 = '7d17806cda7bb4e1c94b6838e52d66c5fa93a421d94d2ada42f23da800b84559'


def load_vendor(tensorlab):
    from mmengine.registry import Registry
    from mmengine.config import ConfigDict

    name = '_flame_ti'
    if name in sys.modules:
        if sys.modules[name].source != str(Path(tensorlab).resolve()):
            raise RuntimeError('A different TI source is already loaded; use a new process')
        return sys.modules[name]
    base = Path(tensorlab) / 'edgeai-mmdetection' / 'mmdet'
    provenance = json.loads((HERE / 'vendor.json').read_text())
    for relative, expected in provenance['files'].items():
        actual = hashlib.sha256((base / relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Pinned TI source mismatch: {relative}')
    for suffix in ['', '.models', '.models.utils', '.models.layers',
                   '.models.backbones', '.models.necks', '.models.dense_heads', '.models.losses']:
        module = types.ModuleType(name + suffix)
        module.__path__ = []
        sys.modules[name + suffix] = module
    root = sys.modules[name]
    root.source = str(Path(tensorlab).resolve())
    registry = types.ModuleType(name + '.registry')
    registry.MODELS = Registry('flame_ti_models')
    sys.modules[registry.__name__] = registry
    utils = types.ModuleType(name + '.utils')
    utils.ConfigType = ConfigDict
    sys.modules[utils.__name__] = utils

    class Imports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'mmcv.cnn' and [a.name for a in node.names] == ['ConvModule']:
                return None  # unused in the official auxiliary head
            if node.module and node.module.startswith('mmdet'):
                node.module = name + node.module[len('mmdet'):]
            return node

    for relative in provenance['files']:
        path = base / relative
        module_name = name + '.' + relative[:-3].replace('/', '.')
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        module.__package__ = module_name.rsplit('.', 1)[0]
        sys.modules[module_name] = module
        tree = Imports().visit(ast.parse(path.read_text(encoding='utf-8')))
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    root.layers = sys.modules[name + '.models.layers.yolo_layers']
    root.geometry = sys.modules[name + '.models.utils.yolo_model_utils']
    root.losses = sys.modules[name + '.models.losses.yolo_loss']
    return root


def lite_surgery(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.MaxPool2d) and isinstance(child.kernel_size, int) and child.kernel_size > 3:
            if child.stride != 1 or child.kernel_size % 2 != 1:
                raise ValueError('Unexpected max-pool for TI Lite surgery')
            setattr(module, name, nn.Sequential(*[
                nn.MaxPool2d(3, 1, 1) for _ in range((child.kernel_size - 1) // 2)]))
        else:
            lite_surgery(child)


class Head(nn.Module):
    def __init__(self, layers, classes):
        super().__init__()
        self.heads = nn.ModuleList([layers.Detection((128, c), classes, reg_max=16)
                                   for c in (128, 192, 256)])

    def forward(self, features):
        return [head(x) for head, x in zip(self.heads, features)]


class Detector(nn.Module):
    def __init__(self, tensorlab, classes=1):
        super().__init__()
        vendor = load_vendor(tensorlab)
        modules = sys.modules
        self.backbone = modules['_flame_ti.models.backbones.yolov9_backbone'].YOLOV9Backbone(
            stem_channels=[32, 64], expand_list=[128, 192, 256],
            pool_kernel_size=3, pool_type='max', init_cfg=None)
        self.neck = modules['_flame_ti.models.necks.yolov9_neck'].YOLOV9Neck(
            in_channels=[128, 192, 256], pool_kernel_size=3, pool_type='max', init_cfg=None)
        self.bbox_head = Head(vendor.layers, classes)
        self.aux_head = modules['_flame_ti.models.dense_heads.yolov9_aux_head'].YOLOV9AuxHead(
            in_channels=[128, 192, 256], num_classes=classes, reg_max=16, init_cfg=None)
        lite_surgery(self)
        self.classes = classes

    @property
    def vendor(self):
        return sys.modules['_flame_ti']

    def forward(self, images, targets=None):
        features = self.backbone(images)
        main = self.bbox_head(self.neck(features))
        if targets is not None:
            aux = self.aux_head(features)[0]
            geometry = self.vendor.geometry.Vec2Box(list(images.shape[-2:]), [8, 16, 32], images.device)
            return self.loss(main, aux, targets, geometry)
        # Raw, pre-NMS outputs: xyxy in input pixels, followed by class probability.
        boxes, scores = [], []
        for output, stride in zip(main, (8, 16, 32)):
            cls, _, distances = output
            h, w = distances.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h, device=images.device),
                                    torch.arange(w, device=images.device), indexing='ij')
            anchors = torch.stack((xx, yy), -1).reshape(1, -1, 2).to(images.dtype) * stride + stride // 2
            distances = distances.flatten(2).transpose(1, 2) * stride
            lt, rb = distances.chunk(2, -1)
            boxes.append(torch.cat((anchors - lt, anchors + rb), -1))
            scores.append(cls.flatten(2).transpose(1, 2).sigmoid())
        return torch.cat((torch.cat(boxes, 1), torch.cat(scores, 1)), -1)

    def loss(self, main, aux, targets, geometry):
        from mmengine.config import ConfigDict
        cfg = ConfigDict(matcher=dict(iou='CIoU', topk=10, factor=dict(iou=6., cls=.5)))
        criterion = self.vendor.losses.YOLOV9Loss(cfg, geometry, class_num=self.classes)
        losses = []
        # Float32 assignment/loss is deliberate, even under mixed precision.
        for outputs in (main, aux):
            cls, anc, boxes = geometry([tuple(t.float() for t in output) for output in outputs])
            aligned, mask = criterion.matcher(targets.float(), (cls.detach(), boxes.detach()))
            target_cls, target_boxes = criterion.separate_anchor(aligned)
            norm = target_cls.sum().clamp_min(1.)
            cls_loss = criterion.cls(cls, target_cls, norm)
            if mask.any():
                weight = target_cls.sum(-1)[mask]
                predicted = (boxes / geometry.scaler[None, :, None])[mask]
                # Vendor BoxLoss computes an NxN IoU matrix then takes its diagonal.
                # Singleton paired dimensions give the SAME CIoU and gradients in O(N).
                paired = self.vendor.geometry.calculate_iou(predicted[:, None, :],
                                                             target_boxes[mask][:, None, :], 'ciou').reshape(-1)
                box_loss = ((1-paired)*weight).sum()/norm
                dfl_loss = criterion.dfl(anc, target_boxes, mask, weight, norm)
            else:
                box_loss = boxes.sum() * 0.
                dfl_loss = anc.sum() * 0.
            losses.append((box_loss, dfl_loss, cls_loss))
        return {name: factor * (losses[0][i] + .25 * losses[1][i])
                for i, (name, factor) in enumerate((('box', 7.5), ('dfl', 1.5), ('cls', .5)))}


def load_pretrained(model, checkpoint):
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest != PRETRAINED_SHA256:
        raise RuntimeError('Pretrained weights do not match the verified official TI checkpoint')
    # This exact official file includes MMEngine metadata, not just tensors.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state.get('model', state))
    expected = model.state_dict()
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    matched = {k: v for k, v in state.items() if k in expected and v.shape == expected[k].shape}
    missing = [k for k in expected if k not in matched]
    allowed = {f'{prefix}.{i}.class_conv.2.{kind}' for prefix in
               ('bbox_head.heads', 'aux_head.heads_aux') for i in range(3) for kind in ('weight', 'bias')}
    if set(missing) - allowed or set(state) - set(expected):
        raise RuntimeError(f'Checkpoint architecture mismatch: missing={missing}, unexpected={set(state)-set(expected)}')
    model.load_state_dict(matched, strict=False)
    if model.classes == 1:
        for head in list(model.bbox_head.heads) + list(model.aux_head.heads_aux):
            nn.init.constant_(head.class_conv[-1].bias, -4.59511985013459)  # .01 prior
    return {'matched_tensors': len(matched), 'reinitialized': missing,
            'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
