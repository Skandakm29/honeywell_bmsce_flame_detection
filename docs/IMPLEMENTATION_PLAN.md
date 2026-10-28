# Implementation plan
Updated 28 October 2026. Completed: full data audit/splits, CPU pilot, tested training/resume pipeline, 20-epoch GPU baseline and verified ONNX export. Current results and exact next experiments: PROJECT_STATUS.md.

Immediate: verify local backup, review full validation failures, then run one evidence-driven improvement at a time. Do not merely extend a plateaued run. Full held-out test and video acceptance follow model/threshold freezing.

| Stage | Scope | State |
|---|---|---|
| Week 1 | Data, baseline detector, reproducible training/export | Baseline complete; improvement/evaluation pending |
| Week 2 | Video inference and temporal alarm logic | Planned |
| Week 3 | Scene calibration / physical scene information | Planned; camera required |
| Week 4 | Obstruction / scene-health detection | Planned |
| Week 5 | Tamper detection | Planned |
| Week 6 | Relay/Ethernet/operator telemetry integration | Planned; interfaces unresolved |
| Week 7 | Electronics / packaging / power | Planned |
| Week 8 | Soak and fault-recovery testing | Planned |
| Week 9 | Representative field evaluation | Planned |
| Week 10 | Handover evidence and deployment documentation | Planned |

Weeks are sequencing from the brief, not verified calendar deadlines. TI compiler/SDK must match eventual hardware; AM62A compatibility, quantization accuracy, 1 TOPS compliance and real-time performance remain unverified. Agree minimum visible flame size, acceptable misses/false alarms and maximum alarm delay before acceptance. Maintain daily contribution notes and weekly team presentations as required by the project brief; never invent contributors.
