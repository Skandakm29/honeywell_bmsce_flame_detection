# Setup and execution
Use Python 3.12 and an isolated environment. Activate .venv/Scripts/activate on Windows cmd or source .venv/bin/activate on Linux. Training was tested on Linux/Colab; native Windows CUDA remains unverified.

```text
python -m venv .venv
# Activate the environment, then choose ONE PyTorch installation:
# GPU, only with a CUDA-12.4-compatible NVIDIA driver:
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# CPU alternative:
# python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/restore_assets.py --handoff PATH_TO_honeywell_training.zip
python scripts/extract_local.py --archive PATH_TO_FASDD_CV.zip
python -m unittest discover -s tests -v
```

Direct requirements include Pillow for audit/tests. Torch and torchvision are installed separately to select the correct CPU/CUDA wheel. In Colab environments lacking ensurepip, use an isolated venv created with with_pip=False and seed pip via the reputable PyPI uv package; do not replace the notebook kernel's Torch. The old upload-specific notebook is intentionally not published as the current execution path. Use ordinary notebook cells to run the commands from this checkout; clone/upload code only, mount Drive for the complete run folder, copy/extract images to VM-local disk. Free GPU availability is conditional. Existing baseline does not need retraining.

```text
# New baseline reproduction ONLY; never overwrite or duplicate an active run:
python scripts/train.py --config configs/full_v1.json
# Continue an interrupted run with identical data/recipe and the entire folder:
python scripts/train.py --config configs/full_v1.json --resume runs/full_v1/last.pth
# Colab: add --output /content/drive/MyDrive/HoneywellTraining/runs/full_v1
```

Before training verify torch.cuda.is_available(), the GPU name and that no other writer owns the output folder. Checkpoint state is portable; CUDA numerics need not match CPU bit-for-bit. Select a new named output/config for any changed experiment. The training script requires --config to avoid accidental pilot/default runs. Its numerical training implementation is unchanged from full_v1. Do not run held-out testing during tuning. Full-validation failure-analysis tooling is a next step, not implemented here.
