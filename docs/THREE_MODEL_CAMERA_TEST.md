# Camera tests — all three flame models

**Goal:** compare YOLOX-Nano-TI-Lite, YOLOX-S-TI-Lite and YOLOv9-S-TI-Lite on the same laptop and the same camera footage. No retraining or board is required.

## 1. Get the files once

Install **Python 3.11** and **Git**. Open PowerShell in the folder where you want the project:

```powershell
git clone --branch feature/yolov9-flame-colab https://github.com/Skandakm29/honeywell_bmsce_flame_detection.git
cd honeywell_bmsce_flame_detection
py -3.11 -m venv .venv-camera
.\.venv-camera\Scripts\python.exe -m pip install numpy opencv-python onnxruntime
New-Item -ItemType Directory -Force models\camera_test, test_clips | Out-Null
```

Obtain these **trained flame ONNX exports**, with their `model_contract.json` and `export.json`, from each model owner. Weights are not in GitHub. Copy/rename the ONNX files to:

| Model | Local filename | Required export |
|---|---|---|
| Nano | `models/camera_test/nano.onnx` | Selected baseline checkpoint, 416 × 416 |
| S | `models/camera_test/s.onnx` | Selected 5+15-epoch checkpoint, 640 × 640 |
| YOLOv9 | `models/camera_test/v9.onnx` | Selected epoch-20 checkpoint, 640 × 640 |

Keep each model's JSON files in its own named subfolder to avoid overwriting them. Check the ONNX hash against its owner's export report. **Do not use COCO weights, a pilot model or an arbitrary last checkpoint.** For YOLOv9, use `outputs/validation/export/flame.onnx` from the shared output folder.

The viewer implements our existing export contracts: YOLOX uses BGR/raw 0–255 and decoded centre/size boxes with objectness × class score; YOLOv9 uses RGB/0–1 and corner boxes with a single class score. Use these exports, not an unrelated YOLO download.

## 2. Check the live webcam

Close Camera/Teams/other apps using the webcam. Run:

```powershell
.\.venv-camera\Scripts\python.exe scripts\live_flame.py --arch yolov9 --model models\camera_test\v9.onnx --source 0
```

Allow camera access for desktop apps in Windows if needed. **Q/Esc:** quit. **S:** save screenshot. Red text means detected; green means no detection above threshold. FPS is processing speed; confidence is a model score, not measured accuracy.

For Nano or S, use `--arch nano --model models\camera_test\nano.onnx` or `--arch s --model models\camera_test\s.onnx` in the same command. All runs default to confidence **0.40**, NMS IoU **0.65**, CPU inference.

If the camera fails to open, append `--backend dshow`. If the wrong camera opens, try `--source 1`, then `--source 2`.

## 3. Record the experiment footage once per camera

Quit the viewer. Open **Windows Camera**, select the desired camera, switch to **Video**, press **Record**, perform one trial, then press **Stop**. Find the saved clip through Camera's gallery and copy it into `test_clips`. Record each trial for **30–60 seconds**. Say what you are doing aloud and note when the actual flame becomes visible.

| Filename example | What to show | Expected result |
|---|---|---|
| `laptop_01_room.mp4` | Ordinary room, face, red/orange clothing or objects | No flame detection |
| `laptop_02_flame.mp4` | Supervised candle in a stable holder; vary camera distance and centre/edge position | Detect visible flame; note small-flame misses |
| `laptop_03_occlusion.mp4` | Partly block the camera's view of the candle with an object safely away from the flame | Detect remaining visible flame where possible |
| `laptop_04_light.mp4` | Phone flashlight at different angles; reflections on glass/metal | No flame detection; this already caused failures in our first test |
| `laptop_05_nuisance.mp4` | Flashing LEDs/vehicle lights; smoke-only or glowing electric heater if safely available | No flame detection; mark unavailable conditions as not tested |
| `laptop_06_screen.mp4` | Another screen displaying a flame photo/video | No physical-flame alarm intended; record any flame detections as screen confusion |

Keep the candle away from people, curtains, paper and electronics; extinguish it afterwards. Do not create smoke, welding or grinding hazards for this test. If welding/grinding can be observed in an existing authorised workplace setup, add a separate clip; otherwise mark them **not tested**. Smoke footage shown on a screen must be labelled replay, not a physical smoke test.

Repeat using the phone camera and name clips `phone_01_room.mp4`, etc. **Android/Windows 11:** Settings → Bluetooth & devices → Mobile devices → link using Link to Windows → enable **Use as connected camera**. **iPhone/Windows:** install the official [iVCam phone app and PC client](https://www.e2esoft.com/ivcam/), connect on the same Wi-Fi and select its virtual camera. [Android requirements and instructions](https://support.microsoft.com/en-gb/windows/apps/phonelink/use-your-mobile-device-s-camera).

## 4. Run the exact same clip through each model

Close Windows Camera. Paste these commands to compare the first clip:

```powershell
.\.venv-camera\Scripts\python.exe scripts\live_flame.py --arch nano --model models\camera_test\nano.onnx --source test_clips\laptop_01_room.mp4 --output results\camera_comparison\nano_laptop_01
.\.venv-camera\Scripts\python.exe scripts\live_flame.py --arch s --model models\camera_test\s.onnx --source test_clips\laptop_01_room.mp4 --output results\camera_comparison\s_laptop_01
.\.venv-camera\Scripts\python.exe scripts\live_flame.py --arch yolov9 --model models\camera_test\v9.onnx --source test_clips\laptop_01_room.mp4 --output results\camera_comparison\v9_laptop_01
```

Repeat for the other clips, changing the source filename and output folder each time. Use a **new output folder** for every run. Let each clip reach its end; **Q** stops it early. Replay processes every decoded frame sequentially and may run slower/faster than real time. Camera mode instead keeps the latest frame. Use recorded replay for comparison.

Each run saves `session.json` (model/source hashes and settings), `frames.csv` (boxes, scores, processing time and source-video time), plus screenshots when you press **S**. Displayed video is not saved automatically. Do not compare laptop and phone results as if they were identical footage; compare all models within each recorded clip.

## 5. Send back a short result table

| Clip/device | Model | Correct flame detections / visible misses | False detections: object + video time | Stable or flickering? | Typical processing FPS |
|---|---|---|---|---|---|

Attach the original clips, run folders and representative screenshots in team Drive. Keep confidence at **0.40** throughout. Do not claim precision/recall or alarm events/hour from unlabelled observations; these are frame detections with no alarm-verification policy. This is one scene/camera trial, not proof across fuels, conditions or AM62A hardware.

**Implementation status:** actual YOLOv9 ONNX inference was checked locally and the laptop webcam test ran successfully. YOLOX adapters follow the repository's export/evaluation code and have synthetic format checks; end-to-end checks with the owners' trained Nano/S files remain required before accepting their results.
