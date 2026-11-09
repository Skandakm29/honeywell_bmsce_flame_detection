"""After a BUG FIX in package code (never rules/config/data): rebuild package_manifest.json and log the change.
  python workflow11\rehash.py "<file changed> | <why> | <what changed>"
"""
import hashlib
import json
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
protected = {'workflow11/config_v11.json', 'PROTOCOL_V11.md'}
old = json.loads((root / 'package_manifest.json').read_text())
files = [q for q in sorted(root.rglob('*')) if q.is_file() and q.relative_to(root).parts[0] != 'inbox'
         and '__pycache__' not in q.parts and q.name not in ('package_manifest.json', 'BUGFIX_LOG.md')]
new = {q.relative_to(root).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in files}
changed = sorted(k for k in new if old.get(k) != new[k])
if protected & set(changed):
    sys.exit(f'REFUSED: protected files changed {sorted(protected & set(changed))}; restore them')
(root / 'package_manifest.json').write_text(json.dumps(new, indent=1))
with (root / 'BUGFIX_LOG.md').open('a') as log:
    log.write(time.strftime('%Y-%m-%d %H:%M') + f" | {changed} | {' '.join(sys.argv[1:])}\n")
print('rehashed; changed:', changed)
