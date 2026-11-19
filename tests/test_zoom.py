import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from detection import ZoomConfig, crop_windows, detect_with_zoom, merge_boxes, select_candidates  # noqa: E402

CFG = ZoomConfig()


def test_candidates_are_weak_boxes_strongest_first_and_capped():
    boxes = [[0, 0, 10, 10, .05], [0, 0, 10, 10, .15], [0, 0, 10, 10, .35], [0, 0, 10, 10, .25],
             [0, 0, 10, 10, .30], [0, 0, 10, 10, .90]]
    c = select_candidates(boxes, CFG)
    assert c[:, 4].tolist() == [.35, .30, .25]          # .90 is already a detection, .05 below candidate_min


def test_crop_windows_stay_inside_and_dedupe():
    cands = [[1900, 1070, 1915, 1078, .3], [1901, 1071, 1916, 1079, .2], [5, 5, 15, 15, .2]]
    w = crop_windows(cands, 1920, 1080, CFG)
    assert w == [(1600, 760, 320, 320), (0, 0, 320, 320)]
    for x0, y0, cw, ch in crop_windows([[0, 0, 4, 4, .3]], 200, 100, CFG):   # frame smaller than crop
        assert (x0, y0, cw, ch) == (0, 0, 200, 100)


def test_merge_keeps_full_and_drops_overlapping_zoom():
    rows, src = merge_boxes([[100, 100, 120, 140, .5]], [[101, 101, 121, 141, .6], [300, 300, 310, 320, .25]])
    assert len(rows) == 2 and src.tolist() == [False, True] and rows[1, 0] == 300
    # a looser crop box around the same flame (low IoU, but centre inside) is a duplicate
    rows, src = merge_boxes([[100, 100, 110, 120, .5]], [[90, 80, 125, 150, .4]])
    assert len(rows) == 1 and src.tolist() == [False]


def test_detect_with_zoom_maps_crop_boxes_back_to_frame():
    frame = np.zeros((1080, 1920, 3), np.uint8)

    def infer(img):
        if img.shape[:2] == (1080, 1920):                 # full frame: one weak candle at (1000, 600)
            return np.array([[990, 580, 1010, 620, .25]])
        return np.array([[150, 140, 170, 180, .30]])      # in the crop, more confident

    rows, src, win = detect_with_zoom(frame, infer, CFG)
    assert win == [(832, 448, 320, 320)]
    assert src.tolist() == [True]
    assert rows[0, :4].tolist() == [982, 588, 1002, 628] and rows[0, 4] == pytest.approx(.30)


def test_zoom_threshold_and_no_candidates():
    frame = np.zeros((480, 640, 3), np.uint8)
    calls = []

    def infer(img):
        calls.append(img.shape)
        return np.array([[10, 10, 30, 30, .50]]) if img.shape == frame.shape else np.array([[1, 1, 5, 5, .15]])

    rows, src, win = detect_with_zoom(frame, infer, CFG)   # strong box only: no crop, one model run
    assert len(calls) == 1 and win == [] and rows.tolist() == [[10, 10, 30, 30, .50]]


def test_config_validation():
    with pytest.raises(ValueError):
        ZoomConfig(candidate_min=.5, full_confidence=.4)
