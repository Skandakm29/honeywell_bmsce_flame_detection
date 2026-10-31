"""Build a portable Colab notebook embedding reviewed code + pinned vendor modules.

Run from the project root after changes. No dataset, credentials, or weights are
embedded. The notebook does not depend on pushing an unreviewed Git branch.
"""
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(tensorlab):
    payload = io.BytesIO()
    vendor = json.loads((HERE/'vendor.json').read_text())
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.iterdir()):
            if path.suffix in ('.py', '.json', '.txt', '.md'):
                archive.writestr('workflow/'+path.name, path.read_bytes().replace(b'\r\n', b'\n'))
        base = Path(tensorlab)/'edgeai-mmdetection/mmdet'
        for relative, expected in vendor['files'].items():
            data = (base/relative).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f'Source hash mismatch: {relative}')
            archive.writestr('tensorlab/edgeai-mmdetection/mmdet/'+relative, data)
        license_path = Path(tensorlab)/'edgeai-mmdetection/LICENSE'
        archive.writestr('tensorlab/LICENSE', license_path.read_bytes())
    encoded = base64.b64encode(payload.getvalue()).decode()
    digest = hashlib.sha256(payload.getvalue()).hexdigest()
    cells = []

    def markdown(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(text).strip().splitlines(True)))

    def code(text):
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                          source=textwrap.dedent(text).strip().splitlines(True)))

    markdown('''
    # TI YOLOv9-S-Lite: flame detection on Colab
    Select **Runtime → Change runtime type → GPU**, then run the cells in order.
    Google Drive authorization is interactive. Use the account selected for this project.

    This notebook embeds the experiment source and six checksum-pinned TI modules.
    No GitHub push, manual labeling, or local dataset upload is required. It can reuse
    the teammate's Drive dataset or download the public Kaggle dataset.

    Three short learning-rate pilots use training data and monitor validation only.
    The best pilot LR starts a fresh full run (up to 40 epochs, EMA, cosine schedule,
    mixed precision, checkpoint recovery). This is a bounded search, not a claim of
    globally optimal hyperparameters. GPU memory determines microbatch size while
    effective batch stays 32. A full-resolution forward/backward/export preflight
    must pass before pilots. The test split is not evaluated by Run all.
    ''')
    code(f'''
    import base64, hashlib, io, json, pathlib, subprocess, sys, zipfile
    from google.colab import drive
    drive.mount('/content/drive')
    payload = base64.b64decode({encoded!r})
    assert hashlib.sha256(payload).hexdigest() == {digest!r}
    ROOT = pathlib.Path('/content/honeywell_yolov9')
    ROOT.mkdir(exist_ok=True)
    zipfile.ZipFile(io.BytesIO(payload)).extractall(ROOT)
    WF = ROOT/'workflow'
    TI = ROOT/'tensorlab'
    RUN = pathlib.Path('/content/drive/MyDrive/HoneywellTraining/experiments/yolov9_s_ti_lite_v1')
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN/'source_bundle.zip').write_bytes(payload)
    print('Experiment:', RUN, '\\nSource bundle SHA256:', {digest!r})
    ''')
    markdown('''
    **Install the isolated, pinned training environment.** This can take several minutes.
    The Colab notebook kernel stays separate; training uses the environment below.
    ''')
    code('''
    ENV = pathlib.Path('/content/yolov9_env_py311')
    # Colab's system Python may omit ensurepip; uv creates/seeds the environment.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.9.5'], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'python', 'install', '3.11'], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--seed', '--allow-existing',
                    '--python', '3.11', str(ENV)], check=True)
    PY = str(ENV/'bin/python')
    def run(*args):
        command = [PY, *map(str, args)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='')
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    run('-m', 'pip', 'install', 'torch==2.5.1', 'torchvision==0.20.1',
        '--index-url', 'https://download.pytorch.org/whl/cu124')
    run('-m', 'pip', 'install', '-r', WF/'requirements.txt', 'kagglehub==0.3.13')
    run('-c', "import torch; assert torch.cuda.is_available(), 'Enable a GPU runtime'; print(torch.__version__, torch.cuda.get_device_name())")
    subprocess.run(['nvidia-smi'], check=True)
    ''')
    markdown('''
    **Obtain the verified pretrained checkpoint and data.** The existing Drive copy is
    preferred when present. Otherwise Kaggle is downloaded automatically. ZIP_PATH is
    optional if dataset.zip already exists in Drive. Originals are preserved.
    ''')
    code('''
    WEIGHTS = ROOT/'yolov9_s_ti_lite_coco.pth'
    DATA = pathlib.Path('/content/yolov9_dataset')
    MANIFESTS = RUN/'manifests'
    DRIVE_DATA = '/content/drive/MyDrive/HoneywellTraining/datasets/fasdd-cv-coco'
    ZIP_PATH = ''  # Optional existing Drive ZIP path; leave empty for Drive folder/Kaggle.
    run(WF/'bootstrap.py', 'weights', '--destination', WEIGHTS)
    command = [WF/'bootstrap.py', 'dataset', '--destination', DATA, '--drive-source', DRIVE_DATA]
    if ZIP_PATH:
        command += ['--archive', ZIP_PATH]
    run(*command)
    run(WF/'data.py', '--root', DATA, '--output', MANIFESTS)
    print((MANIFESTS/'prepare_report.json').read_text()[:5000])
    ''')
    markdown('''
    **Preflight.** Verify source/checkpoint loading, empty-label negatives, gradients,
    metrics, a real 640-pixel training step, and ONNX parity. No full run starts if a
    check fails. Any adaptive batch reduction is recorded before training begins.
    ''')
    code('''
    run(WF/'test_workflow.py', '--tensorlab', TI, '--checkpoint', WEIGHTS, '--export')
    run(WF/'preflight.py', '--tensorlab', TI, '--checkpoint', WEIGHTS,
        '--manifests', MANIFESTS, '--output', RUN/'preflight')
    print((RUN/'preflight/preflight.json').read_text())
    ''')
    markdown('''
    **Train.** Run three LR pilots, select by monitor-validation COCO AP, then train
    the full split from the verified COCO initialization. Rerunning this cell resumes
    matching checkpoints. The source, recipe, manifests, and checkpoint identity must
    match. Keep this cell running; logs/checkpoints are saved to Drive.
    ''')
    code('''
    CONFIG = RUN/'preflight/selected_config.json'
    LOG = RUN/'training_console.log'
    command = [PY, '-u', str(WF/'train.py'), '--config', str(CONFIG),
               '--tensorlab', str(TI), '--checkpoint', str(WEIGHTS),
               '--manifests', str(MANIFESTS), '--output', str(RUN/'training'), '--tune']
    with LOG.open('a', buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='')
            log.write(line)
        status = process.wait()
    if status:
        raise RuntimeError(f'Training stopped (exit {status}). See {LOG}; rerun to resume.')
    ''')
    markdown('''
    **Full validation and ONNX handover.** Produces measured COCO AP50:95, AP50,
    AP75, size-specific AP/AR, and precision/recall/F1 with TP/FP/FN at confidence
    0.40 and matching IoU 0.50. Missing size groups are null, not invented scores.
    Board performance and 1 TOPS compliance remain separate measurements.
    ''')
    code('''
    run(WF/'finish.py', '--tensorlab', TI, '--checkpoint', RUN/'training/full/best.pth',
        '--manifests', MANIFESTS, '--output', RUN/'validation', '--split', 'validation_full')
    print((RUN/'validation/REPORT.md').read_text())
    print('ONNX:', RUN/'validation/export/flame.onnx')
    print('Contract:', RUN/'validation/export/model_contract.json')
    ''')
    markdown('''
    **Optional fixed-model test evaluation is intentionally separate.** The Kaggle
    test has already been examined by the team; it is not a new untouched benchmark.
    After model/recipe/threshold decisions are frozen, use finish.py with
    --split test --final-test --skip-export in a separate cell. Do not tune on it.
    ''')
    notebook = dict(cells=cells, metadata=dict(colab=dict(name='YOLOv9_S_TI_Lite_Flame.ipynb'),
                    accelerator='GPU', kernelspec=dict(display_name='Python 3', name='python3'),
                    language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
    path = HERE/'YOLOv9_S_TI_Lite_Flame.ipynb'
    path.write_text(json.dumps(notebook, indent=1))
    print(path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--tensorlab', required=True)
    build(p.parse_args().tensorlab)
