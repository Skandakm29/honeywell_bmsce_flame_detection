"""Week 4 (flame + obstruction in the same YOLOv9): data, metrics, weight expansion and live-viewer decoding.

  python tests/test_obstruction_class.py -v      (from the repository root)
Model tests need the V10 environment (torch, mmengine, einops, loguru) and are skipped without it.
"""
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/obstruction_w4/workflow_w4'), str(ROOT / 'experiments/yolov9_v10/workflow'),
                str(ROOT / 'experiments/yolov9_v12/workflow12'), str(ROOT / 'scripts')]

import occluders as O  # noqa: E402

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None
HAS_VENDOR = HAS_TORCH and all(importlib.util.find_spec(m) for m in ('mmengine', 'einops', 'loguru'))
HAS_ORT = importlib.util.find_spec('onnxruntime') is not None
HAS_COCO = importlib.util.find_spec('pycocotools') is not None


class OccluderTests(unittest.TestCase):
    def test_spec_is_reproducible_and_label_matches_mask(self):
        a, b = O.sample_spec(random.Random(7)), O.sample_spec(random.Random(7))
        self.assertEqual(a, b)
        json.dumps(a)                                           # specs are stored in manifests
        for seed in range(40):
            spec = O.sample_spec(random.Random(seed))
            box, fraction, mask = O.label(spec, 320, 240, min_fraction=0.)
            ys, xs = np.nonzero(mask)
            self.assertEqual(box, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            self.assertAlmostEqual(fraction, mask.mean())
            if spec['shape'] == 'full':
                self.assertEqual(fraction, 1.)

    def test_render_changes_only_near_the_obstruction(self):
        image = np.full((240, 320, 3), 90, np.uint8)
        spec = dict(O.sample_spec(random.Random(1)), shape='edge', kind='plain', feather=.002,
                    polygon=[[0., 0.], [.3, 0.], [.3, 1.], [0., 1.]])
        out = O.render(image, spec)
        self.assertEqual(out.shape, image.shape)
        self.assertTrue((out[:, 140:] == 90).all())              # far from the edge: untouched
        _, fraction, _ = O.label(spec, 320, 240)
        self.assertAlmostEqual(fraction, .3, places=2)

    def test_hidden_flames_are_dropped_visible_flames_kept(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[:, :50] = 1
        boxes = np.float32([[10, 10, 30, 30], [60, 10, 80, 30], [40, 10, 70, 30]])   # hidden, visible, 1/3 hidden
        self.assertEqual(O.visible_flames(boxes, mask, .5), [1, 2])


@unittest.skipUnless(HAS_TORCH, 'needs torch')
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.records = []
        for i in range(6):
            path = Path(cls.folder.name) / f'{i}.png'
            cv2.imwrite(str(path), np.full((200, 300, 3), 40 * i, np.uint8))
            cls.records.append(dict(image=str(path), width=300, height=200, boxes=[[100., 50., 130., 100.]]))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_synthetic_record_labels_and_hidden_flame(self):
        from data_w4 import OBSTRUCTION, synthetic_record
        spec = dict(O.sample_spec(random.Random(0)), shape='edge', polygon=[[0., 0.], [.5, 0.], [.5, 1.], [0., 1.]])
        record = synthetic_record(self.records[0], spec)
        self.assertEqual(record['labels'], [OBSTRUCTION])           # the flame at x 100-130 is behind the board
        self.assertEqual(record['hidden_flames'], 1)
        for got, want in zip(record['boxes'][0], [0., 0., 150., 200.]):   # polygon rasterisation: within 1 px
            self.assertLessEqual(abs(got - want), 1.)
        self.assertAlmostEqual(record['blind_fraction'], .5, places=2)

    def test_training_targets_keep_class_ids_through_mosaic_and_flip(self):
        from data_w4 import Flames2
        cfg = dict(p=1., min_fraction=.03, hide_if_covered=.5)
        dataset = Flames2(self.records, 640, augment=True, mosaic=.5, occluder_cfg=cfg,
                          augment_cfg=dict(zoom_p=.3, zoom_min=.5, min_box_px=6, paste_p=0., paste_max=1, paste_min_px=8,
                                           paste_max_px=32, resample_p=.1, blur_p=.1, noise_p=.1, jpeg_p=.15))
        seen = set()
        for epoch in range(3):
            dataset.epoch = epoch
            for i in range(len(dataset)):
                tensor, target, _, _ = dataset[i]
                self.assertEqual(tuple(tensor.shape), (3, 640, 640))
                valid = target[target[:, 0] >= 0]
                seen |= set(valid[:, 0].int().tolist())
                self.assertTrue(((valid[:, 1:] >= 0) & (valid[:, 1:] <= 640)).all())
                self.assertTrue((valid[:, 3] > valid[:, 1]).all() and (valid[:, 4] > valid[:, 2]).all())
        self.assertEqual(seen, {0, 1})
        a, b = dataset[2], dataset[2]                                 # deterministic per (seed, epoch, index)
        self.assertTrue((a[0] == b[0]).all() and (a[1] == b[1]).all())

    def test_validation_record_renders_identically_and_without_randomness(self):
        from data_w4 import Flames2, synthetic_record
        record = synthetic_record(self.records[1], O.sample_spec(random.Random(3), dict(shapes={'full': 1})))
        dataset = Flames2([record], 640)
        a, b = dataset[0], dataset[0]
        self.assertTrue((a[0] == b[0]).all())
        self.assertEqual(a[1][:, 0].tolist(), [1.])

    def test_flame_only_records_are_unchanged_without_augmentation(self):
        from data import Flames
        from data_w4 import Flames2
        a, b = Flames(self.records, 640)[0], Flames2(self.records, 640)[0]
        self.assertTrue((a[0] == b[0]).all() and (a[1] == b[1]).all())


@unittest.skipUnless(HAS_TORCH, 'needs torch')
class MetricTests(unittest.TestCase):
    def test_nms_is_per_class(self):
        import torch
        from metrics_w4 import postprocess2
        output = torch.tensor([[10., 10., 100., 100., .9, .05],
                               [12., 12., 100., 100., .05, .8],      # same place, other class: must survive
                               [11., 11., 100., 100., .6, .01]])     # duplicate flame: suppressed
        rows = postprocess2(output, dict(width=640, height=640), np.ones(4), .3)
        self.assertEqual(sorted((r[5], round(r[4], 2)) for r in rows), [(0, .9), (1, .8)])

    def test_blind_share_is_union_area(self):
        from metrics_w4 import blind_share
        self.assertAlmostEqual(blind_share([[0, 0, 50, 100], [25, 0, 75, 100]], 100, 100), .75, places=2)
        self.assertEqual(blind_share([], 100, 100), 0.)

    @unittest.skipUnless(HAS_COCO, 'needs pycocotools')
    def test_obstruction_metrics_split_clean_and_obstructed(self):
        from metrics_w4 import obstruction_metrics
        records = [dict(width=100, height=100, boxes=[[0, 0, 50, 100]], labels=[1], blind_fraction=.5),
                   dict(width=100, height=100, boxes=[], labels=[])]
        detections = [[[0, 0, 50, 100, .9, 1]], [[0, 0, 10, 10, .5, 1]]]
        m = obstruction_metrics(records, detections)
        self.assertEqual(m['image_detection_rate'], 1.)
        self.assertEqual(m['false_obstruction_image_rate'], 1.)
        self.assertAlmostEqual(m['blind_share_mean_abs_error'], 0., places=2)


@unittest.skipUnless(HAS_VENDOR, 'needs the V10 model environment (mmengine, einops, loguru)')
class ExpansionTests(unittest.TestCase):
    def test_grown_model_gives_identical_flame_output(self):
        import torch
        from data import sha256
        from expand_w4 import load_expanded
        from model import TI_COMMIT, Detector
        tensorlab = str(ROOT / 'experiments/yolov9_v10/tensorlab')
        torch.manual_seed(0)
        one, two = Detector(tensorlab, 1).eval(), Detector(tensorlab, 2).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'one.pth'
            torch.save(dict(model=one.state_dict(), epoch=1, identity=dict(ti_commit=TI_COMMIT, config=dict(size=640))), path)
            info = load_expanded(two, path, sha256(path), 640)
        self.assertEqual(len(info['expanded']), 12)
        x = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            a, b = one(x), two(x)
        self.assertTrue(torch.equal(a, b[..., :5]))
        self.assertTrue(torch.allclose(b[..., 5], torch.full_like(b[..., 5], .01), atol=1e-6))


@unittest.skipUnless(HAS_ORT, 'needs onnxruntime')
class LiveViewerTests(unittest.TestCase):
    def test_two_class_decoding_and_blind_percent(self):
        from live_flame import blind_percent, decode_classes
        frame = np.zeros((480, 640, 3), np.uint8)
        rows = np.zeros((1, 4, 6), np.float32)
        rows[0, 0] = [0, 0, 320, 480, .9, .01]                         # flame
        rows[0, 1] = [0, 0, 320, 480, .01, .95]                        # obstruction covering the left half
        rows[0, 2] = [5, 5, 320, 480, .02, .5]                         # duplicate obstruction: NMS
        found = decode_classes(rows, frame, np.ones(4), [.4, .4])
        self.assertEqual(len(found[0]), 1)
        self.assertEqual(len(found[1]), 1)
        self.assertAlmostEqual(blind_percent(found[1], frame), 50., places=0)
        self.assertEqual(blind_percent(np.zeros((0, 5)), frame), 0.)


if __name__ == '__main__':
    unittest.main()
