import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from detection import AlarmRule  # noqa: E402

BOX = [100, 100, 120, 140, .5]


def run(pattern, fps=10, **kw):
    """pattern: list of bools (box present?) per frame; returns states."""
    rule = AlarmRule(**kw)
    return [rule.update(i / fps, [BOX] if p else [])['state'] for i, p in enumerate(pattern)]


def test_steady_flame_alarms_after_enough_hits():
    s = run([True] * 30)
    assert s[0] == 'WARNING'            # one frame is not enough: 1/20 < 0.30
    assert s[5] == 'ALARM'              # 6 hits in a 2 s window at 10 fps = 0.30
    assert s[-1] == 'ALARM'


def test_brief_flicker_never_alarms():
    s = run([True] * 3 + [False] * 40)  # 3 frames = 15% of the window
    assert 'ALARM' not in s and s[0] == 'WARNING' and s[-1] == 'CLEAR'


def test_alarm_holds_three_seconds_then_clears():
    s = run([True] * 20 + [False] * 60)
    last_hit = 19
    on = [i for i, x in enumerate(s) if x == 'ALARM']
    # the window keeps the ratio >= 0.30 for a while after the last hit, then the 3 s off timer runs
    assert on[-1] > last_hit and s[-1] == 'CLEAR'
    assert 3.0 <= (on[-1] + 1 - last_hit) / 10 <= 5.0


def test_two_separate_spots_do_not_add_up():
    rule = AlarmRule()
    far = [800, 800, 820, 840, .5]
    states = [rule.update(i / 10, [BOX] if i % 2 else [far])['state'] for i in range(10)]
    # each spot alone has 5 hits in 10 frames, but the window counts against 20 expected frames: 0.25 < 0.30
    assert 'ALARM' not in states


def test_reads_locked_rule_file(tmp_path):
    p = tmp_path / 'rule.json'
    p.write_text(json.dumps(dict(window_s=2.0, min_hit_ratio=0.3, min_mean_score=0.0, alarm_on_after_s=0.0,
                                 alarm_off_after_s=3.0, tracker=dict(match_iou=0.1, match_centre_x_longest_side=1.5,
                                                                     drop_after_s=1.0))))
    r = AlarmRule.from_json(p)
    assert (r.W, r.h, r.off_s, r.gap, r.match_iou, r.match_centre) == (2.0, 0.3, 3.0, 1.0, 0.1, 1.5)
