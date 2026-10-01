from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
patterns = [
    '*.zip', '*.7z', '*.tar.gz', '*.log', '*.pyc'
]
forbidden_dirs = {'__pycache__', 'outputs_review', '.venv', 'venv', 'env'}
found = []
for p in ROOT.rglob('*'):
    if any(part in forbidden_dirs for part in p.parts):
        if p.is_file():
            found.append(p)
        continue
    if p.is_file() and any(p.match(pattern) for pattern in patterns):
        found.append(p)
if found:
    print('Potentially forbidden/untracked generated files:')
    for p in found[:100]:
        print(' -', p.relative_to(ROOT))
    raise SystemExit(1)
print('PASS: no forbidden archive/cache/log files found in the repository overlay.')
