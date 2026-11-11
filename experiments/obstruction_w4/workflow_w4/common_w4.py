"""Week-4 paths and config. V10 code (common, data, model, metrics, train_v10) is imported read-only from
V10_PACKAGE/workflow; run_w4.py puts it on the path."""
import json
import os
from pathlib import Path

from common import work_dir

HERE = Path(__file__).resolve().parent
PACKAGE_W4 = HERE.parent
CONFIG = json.loads((PACKAGE_W4 / 'config_w4.json').read_text())
V10_PACKAGE = Path(os.environ.get('V10_PACKAGE', PACKAGE_W4.parent / 'yolov9_v10'))
TENSORLAB = V10_PACKAGE / 'tensorlab'
WORK = work_dir()
OUTBOX = WORK / 'outbox'
SOURCE_MANIFESTS = WORK / CONFIG['source_manifests']
MANIFESTS_W4 = WORK / 'manifests_w4'
START = Path(os.environ.get('W4_START_CHECKPOINT', WORK / CONFIG['start']['checkpoint']))
START_SHA256 = os.environ.get('W4_START_SHA256', CONFIG['start']['sha256'])
TRAIN_DIR = WORK / 'train_w4'
SELECT_DIR = WORK / 'select_w4'
EVAL_FILES = ('val.json', 'validation_full.json', 'hn_val.json', 'test.json')
