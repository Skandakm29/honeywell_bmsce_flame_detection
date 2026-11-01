"""Fine-tune integrity, baseline fallback and interrupted recovery checks on CPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import train as training
from data import sha256
from model import TI_COMMIT


class Tiny(torch.nn.Module):
    def __init__(self, unused=None):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 1, 1)

    def forward(self, images, targets):
        return {'cls': self.layer(images).square().mean()}


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = self.root/'sample.png'
        cv2.imwrite(str(image), np.full((16, 32, 3), 114, np.uint8))
        records = [dict(image=str(image), width=32, height=16, boxes=[])]*3
        for name in ('train', 'val'):(self.root/f'{name}.json').write_text(json.dumps(records))
        self.initial = Tiny()
        self.weights = self.root/'initial.pth'
        torch.save(dict(model=self.initial.state_dict(), epoch=20,
                        identity=dict(ti_commit=TI_COMMIT, config=dict(size=32))), self.weights)
        self.cfg = dict(initialization='trained', initial_checkpoint_sha256=sha256(self.weights),
                        seed=42, size=32, batch=2, accumulate=1, workers=0, epochs=3,
                        min_epochs=1, patience=3, weight_decay=.0005, lr=.00001,
                        warmup_epochs=0, warmup_updates=1, min_lr_ratio=.1, amp=False,
                        gradient_clip=10., ema_decay=.9998, checkpoint_every_updates=1,
                        mosaic_probability=0., close_mosaic_epochs=0)

    def tearDown(self):self.temp.cleanup()

    def load(self, path):return torch.load(path, map_location='cpu', weights_only=False)

    def test_strict_trained_loading_and_hash(self):
        model = Tiny()
        training.load_trained(model, self.weights, self.cfg['initial_checkpoint_sha256'], 32)
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,self.initial.state_dict()[key]))
        with self.assertRaises(RuntimeError):training.load_trained(model,self.weights,'wrong',32)
        with self.assertRaises(RuntimeError):training.load_trained(model,self.weights,self.cfg['initial_checkpoint_sha256'],64)
        with self.assertRaises(RuntimeError):training.load_trained(torch.nn.Linear(3,1),self.weights,self.cfg['initial_checkpoint_sha256'],32)

    def test_original_remains_candidate_and_three_epochs_only(self):
        scores = iter([.5, .4, .3, .2])
        def evaluate(*args):return {'coco': {'AP50_95': next(scores)}}, []
        with patch.object(training,'Detector',Tiny),patch.object(training,'evaluate',evaluate):
            training.train(self.cfg,None,self.weights,self.root,self.root/'run',cpu=True)
        best=self.load(self.root/'run/best.pth');last=self.load(self.root/'run/last.pth')
        self.assertEqual(best['epoch'],0);self.assertEqual(last['epoch'],3)
        for key,value in best['model'].items():self.assertTrue(torch.equal(value,self.initial.state_dict()[key]))

    def test_resume_matches_uninterrupted_and_budget_saves(self):
        metric = {'coco': {'AP50_95': .5}}
        def stable(*args):return copy.deepcopy(metric), []
        with patch.object(training,'Detector',Tiny),patch.object(training,'evaluate',stable):
            training.train(self.cfg,None,self.weights,self.root,self.root/'reference',cpu=True)
        calls=0
        def interrupt(*args):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('simulated interruption before epoch validation')
            return stable()
        with patch.object(training,'Detector',Tiny),patch.object(training,'evaluate',interrupt):
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                training.train(self.cfg,None,self.weights,self.root,self.root/'resumed',cpu=True)
        with patch.object(training,'Detector',Tiny),patch.object(training,'evaluate',stable):
            training.train(self.cfg,None,self.weights,self.root,self.root/'resumed',cpu=True)
            training.train({**self.cfg,'max_training_seconds':0},None,self.weights,self.root,self.root/'budget',cpu=True)
        ref=self.load(self.root/'reference/last.pth');res=self.load(self.root/'resumed/last.pth')
        for part in ('model','ema'):
            for key,value in ref[part].items():self.assertTrue(torch.equal(value,res[part][key]),key)
        self.assertEqual(ref['updates'],res['updates'])
        self.assertEqual(json.loads((self.root/'budget/training_summary.json').read_text())['stop_reason'],'time_budget')
        self.assertTrue((self.root/'budget/last.pth').exists())


if __name__=='__main__':unittest.main(verbosity=2)
