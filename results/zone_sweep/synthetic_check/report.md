# Zone threshold sweep

Manifest `configs/zone_sweep_synthetic.json` | 10 labelled clips (5 tune, 5 test) | cost = 10.0 x missed + 1.0 x false alarms + 0.1 x delay s

Chosen on the TUNE clips only; the TEST columns are the honest estimate.

| Setting | grey | zone % | persist s | clear s | tune cost | tune missed | tune false | test cost | test missed | test false | test delay s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best on tune | 25 | 30 | 0.5 | 0.5 | 0.05 | 0 | 0 | 31.05 | 3 | 1 | 0.5 |
| current default | 25 | 25 | 1.5 | 1.0 | 0.15 | 0 | 0 | 31.15 | 3 | 1 | 1.5 |

All 270 combinations: sweep.csv (sorted by tune cost).
