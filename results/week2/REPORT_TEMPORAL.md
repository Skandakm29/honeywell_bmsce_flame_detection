# Temporal alarm rule for V10 (measured on the lab PC; protocol TEMPORAL_PROTOCOL.md)

## Locked rule

```
{
 "model": "V10 (FROZEN_V10/selected.pth, sha256 d046dab6...)",
 "protocol": "TEMPORAL_PROTOCOL.md",
 "score_threshold": 0.4,
 "small_box": {
  "longest_side_px_at_640_below": 16,
  "threshold": 0.25
 },
 "tracker": {
  "match_iou": 0.1,
  "match_centre_x_longest_side": 1.5,
  "drop_after_s": 1.0
 },
 "window_s": 2.0,
 "min_hit_ratio": 0.3,
 "min_mean_score": 0.0,
 "alarm_on_after_s": 0.0,
 "alarm_off_after_s": 3.0,
 "notes": "time based; tested at 10 and 5 fps"
}
```

Small-box threshold search (kept only if it adds no false alarms on the tuning no-fire video):

- small threshold 0.25: false alarms/h with gate 1.52, without 1.52 -> KEPT

## V10 + gate + locked rule (option A)

Rule {'W': 2.0, 'h': 0.3, 'm': 0.0, 'hold': 0.0}, small threshold 0.25.

| set | fps | policy | fires caught | coverage | median s | max s | false alarms / h |
|---|---|---|---|---|---|---|---|
| extra | 10fps | 1_no_temporal_.40 | 33.3% (3) | 15.6% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 10fps | 2_no_temporal_gate | 33.3% (3) | 15.6% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 10fps | 4_final_rule | 33.3% (3) | 23.2% | 6.4 | 6.4 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 1_no_temporal_.40 | 33.3% (3) | 15.4% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 2_no_temporal_gate | 33.3% (3) | 15.4% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 4_final_rule | 33.3% (3) | 23.6% | 6.2 | 6.2 | 43.44 (1 in 0.02 h) |
| onfire_test | 10fps | 1_no_temporal_.40 | 100.0% (6) | 53.1% | 5.5 | 14.3 | 3.27 (16 in 4.90 h) |
| onfire_test | 10fps | 2_no_temporal_gate | 100.0% (6) | 53.1% | 5.5 | 14.3 | 3.27 (16 in 4.90 h) |
| onfire_test | 10fps | 3_old_v10_rule | 66.7% (6) | 51.7% | 6.0 | 11.1 | 2.25 (11 in 4.90 h) |
| onfire_test | 10fps | 4_final_rule | 66.7% (6) | 55.2% | 6.2 | 11.1 | 2.04 (10 in 4.90 h) |
| onfire_test | 5fps | 1_no_temporal_.40 | 100.0% (6) | 53.0% | 7.0 | 14.3 | 3.06 (15 in 4.90 h) |
| onfire_test | 5fps | 2_no_temporal_gate | 100.0% (6) | 53.0% | 7.0 | 14.3 | 3.06 (15 in 4.90 h) |
| onfire_test | 5fps | 4_final_rule | 66.7% (6) | 54.7% | 5.8 | 12.4 | 2.04 (10 in 4.90 h) |
| own | 10fps | 1_no_temporal_.40 | 100.0% (6) | 68.6% | 0.1 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 10fps | 2_no_temporal_gate | 100.0% (6) | 68.6% | 0.1 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 10fps | 3_old_v10_rule | 83.3% (6) | 66.5% | 0.5 | 11.3 | 17.65 (4 in 0.23 h) |
| own | 10fps | 4_final_rule | 83.3% (6) | 72.4% | 0.5 | 10.6 | 13.24 (3 in 0.23 h) |
| own | 5fps | 1_no_temporal_.40 | 100.0% (6) | 68.7% | 0.3 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 5fps | 2_no_temporal_gate | 100.0% (6) | 68.7% | 0.3 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 5fps | 4_final_rule | 83.3% (6) | 72.0% | 0.4 | 12.6 | 13.24 (3 in 0.23 h) |

Per clip, 10 fps (baseline .40 / final rule):

| clip | kind | category | baseline | final |
|---|---|---|---|---|
| extra_corridor_diya_10m | fire | corridor_10m | MISSED | MISSED |
| extra_corridor_diya_1m | fire | corridor_1m | alarm 5.8 s | alarm 6.4 s |
| extra_corridor_diya_4m | fire | corridor_4m | MISSED | MISSED |
| extra_led_candle | no fire | led_fake_flame | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video10.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video15.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video17.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video19.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video20.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video23.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video25.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video27.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video28.mp4 | no fire | onfire_nonfire | 1 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video29.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video30.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video32.mp4 | no fire | onfire_nonfire | 2 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video33.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video35.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video36.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video38.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video39.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video40.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video42.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video43.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video44.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video45.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video47.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video52.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video53.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video57.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video59.mp4 | no fire | onfire_nonfire | 2 false alarms | 2 false alarms |
| videos_TRAINING_SET_1_Video6.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video62.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video63.mp4 | no fire | onfire_nonfire | 1 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video64.mp4 | no fire | onfire_nonfire | 4 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video66.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video67.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video68.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video70.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video71.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video75.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video78.mp4 | no fire | onfire_nonfire | 2 false alarms | 2 false alarms |
| videos_TRAINING_SET_1_Video8.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video80.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video83.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video87.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video88.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video91.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video95.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video96.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video97.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video99.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video156.mp4 | fire | onfire_fire | alarm 4.8 s | alarm 5.5 s |
| videos_TRAINING_SET_1_Video209.mp4 | fire | onfire_fire | alarm 0.0 s | alarm 0.7 s |
| videos_TRAINING_SET_1_Video226.mp4 | fire | onfire_fire | alarm 5.8 s | alarm 6.8 s |
| videos_TRAINING_SET_1_Video234.mp4 | fire | onfire_fire | alarm 5.2 s | MISSED |
| videos_TRAINING_SET_1_Video239.mp4 | fire | onfire_fire | alarm 9.7 s | alarm 11.1 s |
| videos_TRAINING_SET_1_Video75_1.mp4 | fire | onfire_fire | alarm 14.3 s | MISSED |
| ks_screen.mp4 | no fire | screen | 1 false alarms | 1 false alarms |
| km_object.mp4 | no fire | orange | 2 false alarms | 2 false alarms |
| ks_lamp_01.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_lamp_02.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_lamp_03.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_vehicle.mp4 | no fire | vehicle | 0 false alarms | 0 false alarms |
| mm.dark_torch.02.mp4 | no fire | torch | 2 false alarms | 0 false alarms |
| ks_candle_01.mp4 | fire | candle | alarm 0.0 s | alarm 0.5 s |
| ut_diya_02.mp4 | fire | diya | alarm 4.1 s | alarm 10.6 s |
| ut_farflame_03.mp4 | fire | farflame | alarm 58.2 s | MISSED |
| km_mixed.mp4 | fire | mixed_nuisance | alarm 0.1 s | alarm 1.0 s |
| ks_mixed_01.mp4 | fire | mixed_nuisance | alarm 0.0 s | alarm 0.5 s |
| ks_mixed_02.mp4 | fire | mixed_nuisance | alarm 0.0 s | alarm 0.5 s |

False alarms per hour by nuisance type (10 fps):

| type | hours | baseline .40 | final rule |
|---|---|---|---|
| lamp | 0.05 | 0.0 | 0.0 |
| led_fake_flame | 0.01 | 176.3 | 176.3 |
| onfire_nonfire | 4.70 | 3.4 | 2.1 |
| orange | 0.02 | 118.3 | 118.3 |
| screen | 0.02 | 58.4 | 58.4 |
| torch | 0.02 | 112.3 | 0.0 |
| vehicle | 0.02 | 0.0 | 0.0 |

## V10 + gate + rule tuned on own clips only (option B, comparison)

Rule {'W': 3.0, 'h': 0.3, 'm': 0.0, 'hold': 0.0}, small threshold 0.25.

| set | fps | policy | fires caught | coverage | median s | max s | false alarms / h |
|---|---|---|---|---|---|---|---|
| extra | 10fps | 1_no_temporal_.40 | 33.3% (3) | 15.6% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 10fps | 2_no_temporal_gate | 33.3% (3) | 15.6% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 10fps | 4_final_rule | 33.3% (3) | 22.3% | 7.0 | 7.0 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 1_no_temporal_.40 | 33.3% (3) | 15.4% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 2_no_temporal_gate | 33.3% (3) | 15.4% | 5.8 | 5.8 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 4_final_rule | 33.3% (3) | 23.0% | 6.6 | 6.6 | 43.44 (1 in 0.02 h) |
| onfire_test | 10fps | 1_no_temporal_.40 | 100.0% (6) | 53.1% | 5.5 | 14.3 | 3.27 (16 in 4.90 h) |
| onfire_test | 10fps | 2_no_temporal_gate | 100.0% (6) | 53.1% | 5.5 | 14.3 | 3.27 (16 in 4.90 h) |
| onfire_test | 10fps | 3_old_v10_rule | 66.7% (6) | 51.7% | 6.0 | 11.1 | 2.25 (11 in 4.90 h) |
| onfire_test | 10fps | 4_final_rule | 66.7% (6) | 54.6% | 6.5 | 12.3 | 2.04 (10 in 4.90 h) |
| onfire_test | 5fps | 1_no_temporal_.40 | 100.0% (6) | 53.0% | 7.0 | 14.3 | 3.06 (15 in 4.90 h) |
| onfire_test | 5fps | 2_no_temporal_gate | 100.0% (6) | 53.0% | 7.0 | 14.3 | 3.06 (15 in 4.90 h) |
| onfire_test | 5fps | 4_final_rule | 66.7% (6) | 54.5% | 6.3 | 12.8 | 2.04 (10 in 4.90 h) |
| own | 10fps | 1_no_temporal_.40 | 100.0% (6) | 68.6% | 0.1 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 10fps | 2_no_temporal_gate | 100.0% (6) | 68.6% | 0.1 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 10fps | 3_old_v10_rule | 83.3% (6) | 66.5% | 0.5 | 11.3 | 17.65 (4 in 0.23 h) |
| own | 10fps | 4_final_rule | 83.3% (6) | 71.7% | 0.8 | 12.3 | 13.24 (3 in 0.23 h) |
| own | 5fps | 1_no_temporal_.40 | 100.0% (6) | 68.7% | 0.3 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 5fps | 2_no_temporal_gate | 100.0% (6) | 68.7% | 0.3 | 58.2 | 22.06 (5 in 0.23 h) |
| own | 5fps | 4_final_rule | 83.3% (6) | 71.5% | 0.8 | 13.0 | 13.24 (3 in 0.23 h) |

Per clip, 10 fps (baseline .40 / final rule):

| clip | kind | category | baseline | final |
|---|---|---|---|---|
| extra_corridor_diya_10m | fire | corridor_10m | MISSED | MISSED |
| extra_corridor_diya_1m | fire | corridor_1m | alarm 5.8 s | alarm 7.0 s |
| extra_corridor_diya_4m | fire | corridor_4m | MISSED | MISSED |
| extra_led_candle | no fire | led_fake_flame | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video10.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video15.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video17.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video19.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video20.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video23.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video25.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video27.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video28.mp4 | no fire | onfire_nonfire | 1 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video29.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video30.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video32.mp4 | no fire | onfire_nonfire | 2 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video33.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video35.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video36.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video38.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video39.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video40.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video42.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video43.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video44.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video45.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video47.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video52.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video53.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video57.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video59.mp4 | no fire | onfire_nonfire | 2 false alarms | 2 false alarms |
| videos_TRAINING_SET_1_Video6.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video62.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video63.mp4 | no fire | onfire_nonfire | 1 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video64.mp4 | no fire | onfire_nonfire | 4 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video66.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video67.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video68.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video70.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video71.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video75.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video78.mp4 | no fire | onfire_nonfire | 2 false alarms | 2 false alarms |
| videos_TRAINING_SET_1_Video8.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video80.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video83.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video87.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video88.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video91.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video95.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video96.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video97.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video99.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video156.mp4 | fire | onfire_fire | alarm 4.8 s | alarm 5.8 s |
| videos_TRAINING_SET_1_Video209.mp4 | fire | onfire_fire | alarm 0.0 s | alarm 1.0 s |
| videos_TRAINING_SET_1_Video226.mp4 | fire | onfire_fire | alarm 5.8 s | alarm 7.2 s |
| videos_TRAINING_SET_1_Video234.mp4 | fire | onfire_fire | alarm 5.2 s | MISSED |
| videos_TRAINING_SET_1_Video239.mp4 | fire | onfire_fire | alarm 9.7 s | alarm 12.3 s |
| videos_TRAINING_SET_1_Video75_1.mp4 | fire | onfire_fire | alarm 14.3 s | MISSED |
| ks_screen.mp4 | no fire | screen | 1 false alarms | 1 false alarms |
| km_object.mp4 | no fire | orange | 2 false alarms | 2 false alarms |
| ks_lamp_01.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_lamp_02.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_lamp_03.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_vehicle.mp4 | no fire | vehicle | 0 false alarms | 0 false alarms |
| mm.dark_torch.02.mp4 | no fire | torch | 2 false alarms | 0 false alarms |
| ks_candle_01.mp4 | fire | candle | alarm 0.0 s | alarm 0.8 s |
| ut_diya_02.mp4 | fire | diya | alarm 4.1 s | alarm 12.3 s |
| ut_farflame_03.mp4 | fire | farflame | alarm 58.2 s | MISSED |
| km_mixed.mp4 | fire | mixed_nuisance | alarm 0.1 s | alarm 1.3 s |
| ks_mixed_01.mp4 | fire | mixed_nuisance | alarm 0.0 s | alarm 0.8 s |
| ks_mixed_02.mp4 | fire | mixed_nuisance | alarm 0.0 s | alarm 0.8 s |

False alarms per hour by nuisance type (10 fps):

| type | hours | baseline .40 | final rule |
|---|---|---|---|
| lamp | 0.05 | 0.0 | 0.0 |
| led_fake_flame | 0.01 | 176.3 | 176.3 |
| onfire_nonfire | 4.70 | 3.4 | 2.1 |
| orange | 0.02 | 118.3 | 118.3 |
| screen | 0.02 | 58.4 | 58.4 |
| torch | 0.02 | 112.3 | 0.0 |
| vehicle | 0.02 | 0.0 | 0.0 |

## V12 + same locked rule, no gate

Rule {'W': 2.0, 'h': 0.3, 'm': 0.0, 'hold': 0.0}, small threshold None.

| set | fps | policy | fires caught | coverage | median s | max s | false alarms / h |
|---|---|---|---|---|---|---|---|
| extra | 10fps | 1_no_temporal_.40 | 33.3% (3) | 24.3% | 0.4 | 0.4 | 43.44 (1 in 0.02 h) |
| extra | 10fps | 4_final_rule | 33.3% (3) | 23.5% | 6.2 | 6.2 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 1_no_temporal_.40 | 33.3% (3) | 23.9% | 0.4 | 0.4 | 43.44 (1 in 0.02 h) |
| extra | 5fps | 4_final_rule | 33.3% (3) | 23.6% | 6.2 | 6.2 | 43.44 (1 in 0.02 h) |
| onfire_test | 10fps | 1_no_temporal_.40 | 100.0% (6) | 50.0% | 7.9 | 24.3 | 3.27 (16 in 4.90 h) |
| onfire_test | 10fps | 4_final_rule | 66.7% (6) | 54.1% | 6.2 | 12.6 | 2.25 (11 in 4.90 h) |
| onfire_test | 5fps | 1_no_temporal_.40 | 83.3% (6) | 49.9% | 6.0 | 14.5 | 3.06 (15 in 4.90 h) |
| onfire_test | 5fps | 4_final_rule | 66.7% (6) | 53.9% | 6.0 | 13.4 | 2.04 (10 in 4.90 h) |
| own | 10fps | 1_no_temporal_.40 | 100.0% (6) | 69.8% | 0.2 | 55.5 | 22.06 (5 in 0.23 h) |
| own | 10fps | 4_final_rule | 83.3% (6) | 72.8% | 0.5 | 9.4 | 8.83 (2 in 0.23 h) |
| own | 5fps | 1_no_temporal_.40 | 100.0% (6) | 69.8% | 0.3 | 55.6 | 17.65 (4 in 0.23 h) |
| own | 5fps | 4_final_rule | 100.0% (6) | 73.7% | 0.7 | 58.8 | 13.24 (3 in 0.23 h) |

Per clip, 10 fps (baseline .40 / final rule):

| clip | kind | category | baseline | final |
|---|---|---|---|---|
| extra_corridor_diya_10m | fire | corridor_10m | MISSED | MISSED |
| extra_corridor_diya_1m | fire | corridor_1m | alarm 0.4 s | alarm 6.2 s |
| extra_corridor_diya_4m | fire | corridor_4m | MISSED | MISSED |
| extra_led_candle | no fire | led_fake_flame | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video10.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video15.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video17.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video19.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video20.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video23.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video25.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video27.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video28.mp4 | no fire | onfire_nonfire | 1 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video29.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video30.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video32.mp4 | no fire | onfire_nonfire | 2 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video33.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video35.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video36.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video38.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video39.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video40.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video42.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video43.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video44.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video45.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video47.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video52.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video53.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video57.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video59.mp4 | no fire | onfire_nonfire | 2 false alarms | 2 false alarms |
| videos_TRAINING_SET_1_Video6.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video62.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video63.mp4 | no fire | onfire_nonfire | 1 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video64.mp4 | no fire | onfire_nonfire | 4 false alarms | 2 false alarms |
| videos_TRAINING_SET_1_Video66.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video67.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video68.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video70.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video71.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video75.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video78.mp4 | no fire | onfire_nonfire | 2 false alarms | 2 false alarms |
| videos_TRAINING_SET_1_Video8.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video80.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video83.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video87.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video88.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video91.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video95.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_0_Video96.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video97.mp4 | no fire | onfire_nonfire | 0 false alarms | 0 false alarms |
| videos_TRAINING_SET_1_Video99.mp4 | no fire | onfire_nonfire | 1 false alarms | 1 false alarms |
| videos_TRAINING_SET_1_Video156.mp4 | fire | onfire_fire | alarm 4.8 s | alarm 5.5 s |
| videos_TRAINING_SET_1_Video209.mp4 | fire | onfire_fire | alarm 0.0 s | alarm 0.8 s |
| videos_TRAINING_SET_1_Video226.mp4 | fire | onfire_fire | alarm 5.9 s | alarm 6.9 s |
| videos_TRAINING_SET_1_Video234.mp4 | fire | onfire_fire | alarm 24.3 s | MISSED |
| videos_TRAINING_SET_1_Video239.mp4 | fire | onfire_fire | alarm 9.8 s | alarm 12.6 s |
| videos_TRAINING_SET_1_Video75_1.mp4 | fire | onfire_fire | alarm 14.5 s | MISSED |
| ks_screen.mp4 | no fire | screen | 1 false alarms | 0 false alarms |
| km_object.mp4 | no fire | orange | 2 false alarms | 2 false alarms |
| ks_lamp_01.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_lamp_02.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_lamp_03.mp4 | no fire | lamp | 0 false alarms | 0 false alarms |
| ks_vehicle.mp4 | no fire | vehicle | 1 false alarms | 0 false alarms |
| mm.dark_torch.02.mp4 | no fire | torch | 1 false alarms | 0 false alarms |
| ks_candle_01.mp4 | fire | candle | alarm 0.0 s | alarm 0.5 s |
| ut_diya_02.mp4 | fire | diya | alarm 0.8 s | alarm 9.4 s |
| ut_farflame_03.mp4 | fire | farflame | alarm 55.5 s | MISSED |
| km_mixed.mp4 | fire | mixed_nuisance | alarm 0.3 s | alarm 1.0 s |
| ks_mixed_01.mp4 | fire | mixed_nuisance | alarm 0.0 s | alarm 0.5 s |
| ks_mixed_02.mp4 | fire | mixed_nuisance | alarm 0.0 s | alarm 0.5 s |

False alarms per hour by nuisance type (10 fps):

| type | hours | baseline .40 | final rule |
|---|---|---|---|
| lamp | 0.05 | 0.0 | 0.0 |
| led_fake_flame | 0.01 | 176.3 | 176.3 |
| onfire_nonfire | 4.70 | 3.4 | 2.3 |
| orange | 0.02 | 118.3 | 118.3 |
| screen | 0.02 | 58.4 | 0.0 |
| torch | 0.02 | 56.1 | 0.0 |
| vehicle | 0.02 | 44.3 | 0.0 |

