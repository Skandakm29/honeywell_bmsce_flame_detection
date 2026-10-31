"""Contract tests and optional actual-model/ONNX checks; no trained-metric claims."""
import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from data import Flames, collate, letterbox
from metrics import coco_metrics, operating_metrics, postprocess
from model import Detector, load_pretrained

ARGS = None


class DataMetricsTests(unittest.TestCase):
    def test_ema_optimization_is_bitwise_equivalent(self):
        from train import update_ema
        model = torch.nn.Sequential(torch.nn.Linear(7, 7), torch.nn.BatchNorm1d(7))
        reference, optimized = copy.deepcopy(model), copy.deepcopy(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.123)
            model[1].num_batches_tracked.add_(3)
            for decay in (0., .25, .9998):
                for key, value in reference.state_dict().items():
                    current = model.state_dict()[key].detach()
                    if value.is_floating_point():
                        value.mul_(decay).add_(current, alpha=1-decay)
                    else:
                        value.copy_(current)
                update_ema(optimized, model, decay)
                for key, value in reference.state_dict().items():
                    self.assertTrue(torch.equal(value, optimized.state_dict()[key]), key)

    def test_resize_and_postprocess_roundtrip(self):
        image = np.full((73, 201, 3), (10, 20, 30), np.uint8)
        boxes = [[3., 5., 190., 67.]]
        canvas, scaled, scale = letterbox(image, boxes, 64)
        output = torch.tensor([scaled[0].tolist()+[.9]])
        restored = postprocess(output, dict(width=201, height=73), scale)
        np.testing.assert_allclose(restored[0][:4], boxes[0], atol=1e-4)
        self.assertEqual(canvas[63, 0].tolist(), [114, 114, 114])

    def test_rgb_normalization_and_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'image.png'
            cv2.imwrite(str(path), np.full((32, 64, 3), (10, 20, 30), np.uint8))
            r = dict(image=str(path), width=64, height=32, boxes=[])
            item = Flames([r], 64)[0]
            np.testing.assert_allclose(item[0][:, 0, 0], np.array([30, 20, 10])/255., atol=1e-7)
            self.assertEqual(item[1].tolist(), [[-1., 0., 0., 0., 0.]])
            self.assertEqual(tuple(collate([item, item])[1].shape), (2, 1, 5))

    def test_duplicate_detections_are_false_positives(self):
        records = [dict(boxes=[[0, 0, 10, 10]]), dict(boxes=[])]
        predictions = [dict(image_id=0, bbox=[0, 0, 10, 10], score=.9),
                       dict(image_id=0, bbox=[0, 0, 10, 10], score=.8),
                       dict(image_id=1, bbox=[0, 0, 10, 10], score=.7)]
        m = operating_metrics(records, predictions)
        self.assertEqual((m['TP'], m['FP'], m['FN']), (1, 2, 0))

    def test_coco_perfect_and_empty_predictions(self):
        records = [dict(width=64, height=64, boxes=[[0, 0, 10, 10]])]
        perfect = [dict(image_id=0, category_id=1, bbox=[0, 0, 10, 10], score=.9)]
        self.assertAlmostEqual(coco_metrics(records, perfect)['AP50_95'], 1.)
        self.assertEqual(coco_metrics(records, [])['AP50_95'], 0.)


@unittest.skipUnless('--tensorlab' in sys.argv, 'requires pinned TI source')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = Detector(ARGS.tensorlab)

    def test_negative_and_mixed_batches_have_finite_loss_and_gradients(self):
        for positive in (False, True):
            model = self.model.train()
            model.zero_grad(set_to_none=True)
            target = torch.tensor([[[-1., 0, 0, 0, 0]], [[-1., 0, 0, 0, 0]]])
            if positive:
                target[0, 0] = torch.tensor([0., 8, 8, 40, 40])
            components = model(torch.rand(2, 3, 64, 64), target)
            loss = sum(components.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_aligned_ciou_matches_vendor_values_and_gradients(self):
        vendor = self.model.vendor.geometry
        a = torch.tensor([[1., 2, 8, 9], [4., 3, 16, 18]], requires_grad=True)
        b = torch.tensor([[2., 1, 9, 10], [5., 5, 17, 20]])
        pairwise = vendor.calculate_iou(a, b, 'ciou').diag()
        aligned = vendor.calculate_iou(a[:, None], b[:, None], 'ciou').reshape(-1)
        torch.testing.assert_close(pairwise, aligned)
        torch.testing.assert_close(torch.autograd.grad(pairwise.sum(), a, retain_graph=True)[0],
                                   torch.autograd.grad(aligned.sum(), a)[0])

    def test_pinned_pretrained_and_ema_copy(self):
        if not ARGS.checkpoint:
            self.skipTest('no pretrained checkpoint supplied')
        result = load_pretrained(self.model, ARGS.checkpoint)
        self.assertEqual(len(result['reinitialized']), 12)
        self.assertGreater(result['matched_tensors'], 1700)
        copy.deepcopy(self.model)

    def test_onnx_export_and_parity(self):
        if not ARGS.export:
            self.skipTest('pass --export for ONNX integration check')
        from finish import export
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.png'
            cv2.imwrite(str(path), np.random.default_rng(42).integers(0, 256, (51, 101, 3), dtype=np.uint8))
            dataset = Flames([dict(image=str(path), width=101, height=51, boxes=[])], 64)
            report = export(self.model.eval(), dataset, directory)
            self.assertTrue(report['parity_passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorlab')
    parser.add_argument('--checkpoint')
    parser.add_argument('--export', action='store_true')
    ARGS, remainder = parser.parse_known_args()
    # Keep original argv until decorators have seen it; tests only receive remaining flags.
    unittest.main(argv=[sys.argv[0]]+remainder, verbosity=2)
