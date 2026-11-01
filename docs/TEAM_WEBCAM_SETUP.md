# Test your flame model on your laptop webcam

Each teammate tests **their own trained model**. You do not need the other two models to start. Use the same six scenarios below; later, replay shared recordings through all three for a fair comparison.

## 1. Give this prompt to your AI

Copy the following text into your AI and attach this document, your `model_contract.json` and `export.json`. Tell it where your downloaded `flame.onnx` is.

```text
Help me set up a laptop webcam experiment for my existing trained flame detector.
I am one of three teammates, each testing our own model. Ask which model I own
and which operating system I use if I have not said. I will provide flame.onnx,
model_contract.json and export.json, plus their local folder path.

Repository: https://github.com/Skandakm29/honeywell_bmsce_flame_detection
Branch: feature/yolov9-flame-colab
Viewer: scripts/live_flame.py

Use the existing viewer and a separate Python environment with numpy,
opencv-python and onnxruntime. First inspect my model contract and check its
ONNX hash against export.json. Check that its input/output format matches the
selected adapter. Use --arch nano for YOLOX-Nano-TI-Lite (416), --arch s for
YOLOX-S-TI-Lite (640), or --arch yolov9 for YOLOv9-S-TI-Lite (640).
Do not substitute a COCO model. Do not retrain or install the training stack.
If the contract differs, investigate the mismatch instead of guessing.

Give me the exact commands for my machine and my actual model path. Start at
confidence 0.40 on CPU. Show live boxes, scores and processing FPS. Explain
that Q/Esc quits and S saves a screenshot when the viewer window is focused.
Help with camera permissions/index/backend if needed. The viewer saves CSV
logs and screenshots under results/webcam. Full video is not recorded by it.

Guide me through the six trials in this document, one at a time. Help me record
failures honestly. Do not claim accuracy/recall or alarm rates without labelled
evidence. No purchases, cloud training or account sign-ins are needed for the
laptop webcam test. For phone cameras, follow the optional phone section.
```

## 2. Simple Windows setup

Install Python 3.11 and Git if needed. Download **your model's three files** from the shared Drive into one local folder. Open PowerShell in a folder where you want to clone the project:

```powershell
git clone --branch feature/yolov9-flame-colab https://github.com/Skandakm29/honeywell_bmsce_flame_detection.git
cd honeywell_bmsce_flame_detection
py -3.11 -m venv .venv-camera
.\.venv-camera\Scripts\python.exe -m pip install numpy opencv-python onnxruntime
```

Already cloned? Use that project folder and ask your AI to update the branch without overwriting local changes. On macOS/Linux, ask the AI to adapt the environment commands to your OS.

Close other camera apps. Run **only the command for your model**, replacing the quoted path with your actual ONNX path:

```powershell
# Nano owner
.\.venv-camera\Scripts\python.exe scripts\live_flame.py --arch nano --model "C:\YOUR_FOLDER\Nano\flame.onnx" --source 0

# S owner
.\.venv-camera\Scripts\python.exe scripts\live_flame.py --arch s --model "C:\YOUR_FOLDER\S\flame.onnx" --source 0

# YOLOv9 owner
.\.venv-camera\Scripts\python.exe scripts\live_flame.py --arch yolov9 --model "C:\YOUR_FOLDER\YOLOv9\flame.onnx" --source 0
```

If the wrong camera opens, quit and try `--source 1`, then `--source 2`. If it cannot open, check Windows camera access for desktop apps, close Camera/Teams, and try appending `--backend dshow` on Windows.

**Buttons:** click the viewer window, then **S** to save a screenshot; **Q** or **Esc** to quit. Red text means detections; green means no detection above 0.40. The confidence threshold stays fixed; the displayed score belongs to a detected box. These are frame detections, not verified alarms.

## 3. Do these six trials

Spend roughly 30–60 seconds per trial. Save at least one screenshot of each useful success/failure. Record approximate distance and lighting; do not invent distances from the image.

| Trial | Show the camera | Check |
|---|---|---|
| 1. Normal scene | Room, person, red/orange clothing or objects | No flame boxes expected |
| 2. Actual flame | A supervised candle in a stable holder; vary camera distance and centre/edge position | Does it detect the visible flame, including smaller views? |
| 3. Partial obstruction | Partly block the camera's view of the candle, keeping the object safely away from it | Does it detect the remaining visible flame? A fully hidden flame cannot be expected to be seen |
| 4. Flashlight/reflections | Phone flashlight at different angles; sunlight or reflections on metal/glass when available | No flame boxes expected; the flashlight caused false detections in our first YOLOv9 trial |
| 5. Other distractions | Flashing LEDs/vehicle lights; smoke without flame or a glowing electric heater if safely available | No flame boxes expected. Actual burner flames are still flames |
| 6. Screen replay | Phone/monitor showing a flame photo or video | Record detections as screen confusion under our intended physical-flame policy |

Keep the candle away from curtains, paper, people and electronics; extinguish it afterwards. Do not create smoke, welding or grinding hazards for testing. The mentor also requested welding and grinding sparks: use existing authorised workplace conditions if available; otherwise mark them **not tested**. Label smoke footage on a screen as replay.

## 4. Save and send results

The program prints the output directory, normally `results/webcam/<session_time>/`. Keep its `session.json`, `frames.csv` and screenshots. Make a small table:

| Model/camera | Trial | What happened / approximate video time if recorded | Miss or false detection? | Stable or flickering? | Processing FPS |
|---|---|---|---|---|---|

Upload the session folder and table to team Drive. No need to upload your Python environment. These observations do not give a measured precision/recall score without ground-truth labels.

For an exact three-model comparison, quit the viewer and use Windows Camera to record each trial once. Copy the clips locally and replace `--source 0` with `--source "C:\path\to\trial.mp4"` in the same model command. Every decoded frame is processed in order; replay speed may differ from real time. Source-video times are saved in the CSV. Share those exact clips so each teammate runs identical footage. Compare speed only on the same laptop/runtime.

## 5. Optional phone camera

The phone supplies images; inference still runs on the laptop.

- **Android + Windows 11:** Settings → Bluetooth & devices → Mobile devices → link phone using Link to Windows → enable **Use as connected camera** and allow camera permissions. Then try the new camera index. [Microsoft setup/requirements](https://support.microsoft.com/en-gb/windows/apps/phonelink/use-your-mobile-device-s-camera).
- **iPhone + Windows:** use the official [iVCam phone app and Windows client](https://www.e2esoft.com/ivcam/) on the same Wi-Fi. Once its preview works, choose its camera index in the viewer. Check current app terms/restrictions before installing.

## What has been checked

YOLOv9 was run on the project owner's actual laptop webcam and its exported model passed replay checks. Nano/S preprocessing and decoding follow the repository's existing exports and passed synthetic checks. Their actual trained ONNX files still need an initial run on each teammate's system. Camera drivers, permissions and model-file differences can require setup fixes; identical behaviour across laptops is not guaranteed.
