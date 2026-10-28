# Algorithm decision record

Updated: 28 October 2026 | Current baseline: **TI YOLOX-Nano-Lite**

## Decision and scope

Choose Nano-Lite as the first architecture to train for **visible flame localization**, based chiefly on its documented TI deployment route. It is a general detector, not a ready-made flame detector. Generic pretrained weights do not establish flame capability [R1–R3].

Intended output: flame boxes and scores; smoke-only scenes are negatives. Annotation details and camera spectrum still need confirmation. A confidence score is not automatically a calibrated probability. This choice does not imply all-fuel, thermal, or FS24X Plus sensing performance.

The choice is implemented in the laptop pilot and pinned to TI source cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6. Full data preparation and the 20-epoch Colab T4 baseline are complete. Current status/results: [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Evidence and alternatives

This is a selective survey. Source facts support the descriptions below; selection reasons are **our engineering judgments**. No controlled project experiment ranks these methods.

| Considered approach | Verified basis | Why selected or deferred |
|---|---|---|
| Colour/motion/flicker | TÃ¶reyin combines colour, motion and wavelet-based temporal/spatial cues [R4]. | Keep relevant to later flicker checks; prefer learned localization as the main baseline. No measured claim that classical methods are inferior here. |
| Whole-image CNN classifier | CNN-based surveillance fire detection is established [R5]. | A whole-image label alone does not provide boxes. Prefer direct detection; this is a task choice, not a claim that all CNN methods lack localization. |
| SSD / MobileNetV2-SSDLite | Single-stage boxes and efficient mobile features [R6, R7]. | Credible comparison candidate; defer behind the specific TI Nano-Lite application. |
| Faster R-CNN | Shared features, region proposals and refinement [R8]. | Defer its two-stage integration for the initial baseline. Not proven too slow or incompatible. |
| YOLOv8n | Official small detector and export documentation; no formal YOLOv8 architecture paper [R9]. | Strong alternative; prefer the documented TI Lite route. AGPL/Enterprise options need project review, not assumptions about legal suitability. |
| DSS-YOLO / improved YOLOv5s | Published changes for small/obscured fire targets [R10, R11]. | Establish a standard baseline before custom modules. Compatibility and reproduction need checking; these are not proven inferior. |
| RT-DETR | Transformer detector avoiding duplicate-box suppression (NMS) [R14]. | Defer until target measurements justify it; GPU speed does not establish 1 TOPS performance. |
| Detector + LSTM | Suspected regions processed across video frames [R15]. | Relevant later; sequence training/evaluation adds a separate video-data task. |
| **YOLOX-Nano-Lite** | YOLOX architecture and a specific TI AM62A demonstration/export route [R1–R3]. | **Current choice:** localization with directly relevant vendor integration evidence. Flame baseline measured; no same-data comparison against alternatives. |

## Why this choice is defensible

- **Relevant deployment precedent:** TI selects a 416 × 416 Nano-Lite model for AM62A defect detection [R2]. This reduces uncertainty; it does not verify our future export.
- **Export path:** TI documents ONNX plus TIDL detector metadata [R3]. Extra deployment artifacts may accompany the delivered ONNX model.
- **Appropriate output:** object boxes support initial localization [R1]. Physical distance still requires calibration.
- **Reproducible starting point:** use a pinned reference configuration before modifying architecture. This is our proposed development strategy, not a benchmark result.
- **Trade-off:** The full baseline ran on Colab T4; lab RTX 5000 execution and target SDK compatibility remain unverified.

The verified advantage is the vendor route, **not superior flame accuracy**. Absence of an equally detailed deployment review for an alternative is not proof that it is unsupported.

## Numbers and limits

| Published evidence | Correct interpretation |
|---|---|
| Original YOLOX-Nano: 0.91M parameters, 1.08 GFLOPs at 416 × 416 [R16] | Upstream model only; not a measured specification of TI Lite. |
| TI Nano-Lite: 8.88 ms/frame, Table 3-1 [R2] | Historical AM62A configuration; not our flame model, full alarm delay or 1 TOPS result. |
| TI: YOLOX-Nano 8.69 ms at 416; SSD-MobileNetV2 14.51 ms at 512 on AM62A7 [R12] | Different configurations/resolutions; not a controlled architectural ranking or the exact SSDLite comparison. |

Do not equate FLOPs, MACs and integer TOPS. Keep the project's **1 TOPS** ceiling. COCO results do not predict flame accuracy; differing datasets, metrics and hardware prevent casual cross-paper ranking [R13].

Revisit the choice if training/export/compilation is impractical, hardware changes, or measured misses, false alarms, quantization loss or latency are unacceptable. Test small/distant flames and confusing negatives. Acceptable numerical targets remain undecided.

## Sources and review depth

Reviewed in the 27 October 2026 conversation. Pin mutable code revisions during implementation. Abstract screening is not a full methodological audit; nothing has been independently reproduced.

| ID | Source | Review basis |
|---|---|---|
| R1 | Ge et al., [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430) (2021) | Author technical-report abstract and repository. |
| R2 | TI, [Vision AI-Based Defect Detection on AM62A](https://www.ti.com/lit/an/spradc9/spradc9.pdf) (2023) | PDF sections 3.1–3.2 and Table 3-1. |
| R3 | TI, [EdgeAI-MMDetection](https://github.com/TexasInstruments/edgeai-tensorlab/blob/main/edgeai-mmdetection/README.md) | Environment, Models and Export sections. |
| R4 | TÃ¶reyin et al., [Computer vision based method for real-time fire and flame detection](https://doi.org/10.1016/j.patrec.2005.06.015) (2006) | Publisher abstract/algorithm snippets; later DOI fetch failed. |
| R5 | Muhammad et al., [CNNs Based Fire Detection in Surveillance Videos](https://doi.org/10.1109/ACCESS.2018.2812835) (2018) | Abstract/index screening; direct publisher access blocked. |
| R6 | Liu et al., [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) (2016) | Author abstract. |
| R7 | Sandler et al., [MobileNetV2](https://arxiv.org/abs/1801.04381) (2018) | Original paper/abstract information. |
| R8 | Ren et al., [Faster R-CNN](https://arxiv.org/abs/1506.01497) (2015) | Author abstract. |
| R9 | [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) | Official software/model documentation. |
| R10 | Wang et al., [DSS-YOLO](https://www.nature.com/articles/s41598-025-93278-w) (2025) | Article abstract and conclusion/limitations. |
| R11 | Lv et al., [Small-target fire detection based on YOLOv5s](https://www.nature.com/articles/s41598-024-64934-4) (2024) | Publisher abstract; full-page recheck failed. |
| R12 | [TI object-detection benchmarks](https://www.ti.com/technologies/edge-ai/use-cases/object-detection-vision-systems.html) | Vendor table. |
| R13 | Huang et al., [Speed/accuracy trade-offs](https://arxiv.org/abs/1611.10012) (2017) | Author abstract. |
| R14 | Zhao et al., [DETRs Beat YOLOs](https://arxiv.org/abs/2304.08069) (CVPR 2024) | Author abstract. |
| R15 | [A Video-Based Fire Detection Using Deep Learning Models](https://www.mdpi.com/2076-3417/9/14/2862) (2019) | Publisher abstract; full-page access rate-limited. |
| R16 | [Original YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) | Light Models table, export support and licence label. |
| R17 | Wang et al., [An open flame and smoke detection dataset for deep learning in remote sensing based fire detection](https://doi.org/10.1080/10095020.2024.2347922) | Published online 2024; Geo-Spatial Information Science 28(2), 511–526 (2025). Author/title/issue metadata reverified on publisher platform SciOpen. |

Four shareable papers reverified on 27 October: R1 (YOLOX architecture, arXiv technical report), R4 (classical flame/flicker, peer-reviewed journal), R10 (lightweight fire-specific alternative, journal), and R17 (our dataset, journal). R4's metadata/abstract were reverified through the [author's university record](https://research.itu.edu.tr/tr/publications/computer-vision-based-method-for-real-time-fire-and-flame-detecti/); direct publisher access was blocked. These are verified publications, not independently reproduced experiments.

The YOLOX paper describes the base architecture, not TI's modifications. Our actual pilot code comes from the [pinned TI fork](https://github.com/TexasInstruments/edgeai-yolox/tree/cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6), now archived; its listed successor is edgeai-tensorlab. TI's AM62A application note R2 is deployment evidence, not a flame-specific research paper. None of these references proves our chosen model beats alternatives on flames.

## History

- 2026-10-27: Recommended Nano-Lite, then documented the user's current baseline direction.
- 2026-10-27: Condensed record and linked the proposed full plan. Selection unchanged; all measured claims remain pending.
- 2026-10-27: FASDD_CV V9 became the preferred dataset to audit; hardware findings informed the host recommendation. Algorithm choice unchanged. Current handoff: [PROJECT_STATUS.md](PROJECT_STATUS.md).
- 2026-10-27: Implementation authorized on laptop. Pinned official legacy TI YOLOX source cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6, configuration exps/default/yolox_nano_ti_lite.py and its matching TI COCO checkpoint. Confirmed Lite uses regular convolutions, ReLU, convolutional input stem and split max-pooling. CPU loop loads these unchanged 2D modules without unrelated pose/CUDA training infrastructure; exact target SDK compatibility remains pending. This is an implementation-route change from the proposed MMDetection stack, not a switch to upstream Nano.
- 2026-10-27: First five-epoch local pilot and ONNX parity completed. Measured one-class model has 2,241,998 parameters; do not assign upstream Nano or historical table figures to it. Weak pilot AP50:95=0.0682 does not establish superiority or justify switching architectures yet; expand data/training and inspect failures first. Model choice retained; full results in PROJECT.md.
- 2026-10-27: Full archive verified and overlap-screened. Added reference EMA and warmup/cosine scheduling, modest exposure/colour variation, gradient accumulation and checkpoint contracts; architectural choice unchanged. Same-CPU resume test passed. Main GPU recipe has not been trained yet; no accuracy/production claims made from these code changes.

For future changes record the date, new evidence, revised decision and limitations; preserve why earlier alternatives were deferred.


- 2026-10-27: Full_v1 completed 20 epochs on Colab T4. Best epoch 17 monitor AP50:95 0.32925 / AP50 0.62687; ONNX checker and parity passed. Choice retained, with substantial missed-detection/generalization work still required. No competing detector was trained on the same split, so this does not demonstrate superiority; numerical production acceptance and TI hardware validation remain pending. Evidence: ../results/full_v1_summary.json.
