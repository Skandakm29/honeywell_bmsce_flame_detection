# Laptop and phone camera test

This viewer runs our trained YOLOv9 ONNX on the **laptop CPU**. It displays boxes and confidence scores. It does not train the model or raise verified alarms. It keeps the latest available camera frame to avoid accumulating a queue while processing.

## Start on the project owner's laptop

Close Windows Camera, Teams and other apps using the camera. Open PowerShell:

```powershell
Set-Location E:\Projects\honeywell\honeywell_bmsce_flame_detection
.\.venv-yolov9\Scripts\python.exe .\scripts\live_flame.py --source 0
```

Allow camera access if prompted. In Windows settings, camera access for desktop apps must be enabled. The viewer uses the verified local `results/lab_backup_20261031/outputs/validation/export/flame.onnx`.

Press **Q** or **Esc** to quit; **S** saves a screenshot. Logs and manually saved screenshots go into a new directory under `results/webcam/`. Full video is not recorded automatically.

If the wrong camera opens, quit and change `--source 0` to `--source 1`, then `--source 2` if needed. Indices depend on the PC. If opening fails, also try `--backend dshow` or `--backend msmf`; virtual camera support depends on its driver.

## Android phone — easiest first choice on Windows 11

1. Put the phone and PC on the same Wi-Fi network.
2. On Windows: **Settings → Bluetooth & devices → Mobile devices → Manage devices**.
3. Add/link the Android phone using **Link to Windows** and approve camera permissions.
4. Enable **Use as connected camera** for that phone. Follow any Windows update/install prompts for the feature.
5. Rerun the viewer with the phone's camera index, usually a different index from the laptop camera. Choose the rear camera on the phone if offered.

Microsoft lists Windows 11+, Android 10+ and Link to Windows 1.24022.0+ for this feature; check its current requirements if the toggle is missing:
<https://support.microsoft.com/en-gb/windows/apps/phonelink/use-your-mobile-device-s-camera>

## iPhone — requires a webcam bridge on Windows

1. Install **iVCam** from the developer's official links on both the iPhone and Windows PC: <https://www.e2esoft.com/ivcam/>.
2. Connect both devices to the same Wi-Fi network. Open both apps and allow camera/local-network permissions.
3. Wait until the phone picture appears in the Windows iVCam client.
4. Run our viewer using the new virtual camera index. The phone supplies video; inference still runs on the laptop.

Developer's Wi-Fi instructions: <https://www.e2esoft.com/ivcam-connect-via-wifi/>. Check current app terms/restrictions before installing; the model does not require a paid subscription.

## Quick testing procedure

Keep confidence at **0.40** during the first comparison. Use the same scenes for each camera.

| Trial | What to do | What to record |
|---|---|---|
| Normal room | Observe the room, clothes and ordinary objects for 2 minutes | Unexpected boxes and whether they persist; save representative screenshots |
| Confusing objects | Show red/orange objects, lamps and reflections for 2 minutes | Which objects trigger detections |
| Screen replay | Show existing flame footage on another screen for 1 minute | Whether the viewer processes it and draws boxes. Label this **screen replay**, not a physical-flame test |
| Normal camera movement | Move the camera gently and vary room lighting for 1 minute | Flicker, unstable boxes, processing rate and disconnections |

Do not light a fire for this setup check. The mentor lists screens showing flames as a nuisance condition: detection of displayed flames establishes that inference works, but is a false positive for the intended physical-flame alarm policy. Rejection of a replay also does not establish inability to detect a physical flame.

For each camera, note device, lighting, test duration, observed false detections and displayed processing FPS. Confidence is a model score, not a measured percentage accuracy. A slower CPU processing rate does not necessarily mean the camera itself is slow. This demo does not measure true sensor-to-display latency.

The CSV stores detections on processed frames; it cannot produce recall or false alarm events/hour without ground truth and a defined alarm policy. Frame counts also cannot be compared directly between cameras with different processing rates. A few minutes of testing is a setup check, not a reliability claim.

## Another laptop

Copy the ONNX file, its `model_contract.json` and `scripts/live_flame.py`. Install Python 3.11, then create a small environment in the project folder:

```powershell
py -3.11 -m venv .venv-webcam
.\.venv-webcam\Scripts\python.exe -m pip install numpy opencv-python onnxruntime
.\.venv-webcam\Scripts\python.exe .\scripts\live_flame.py --model "C:\path\to\flame.onnx" --source 0
```

The full training environment, dataset and TI source tree are not needed for this ONNX viewer.
