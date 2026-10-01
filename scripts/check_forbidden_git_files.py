from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
patterns = ['*.zip', '*.7z', '*.tar.gz', '*.log', '*.pyc']
forbidden_dirs = {'__pycache__', 'outputs_review', '.venv', 'venv', 'env'}
skip_dirs = {'.git'}
found = []

for p in ROOT.rglob('*'):
    rel_parts = p.relative_to(ROOT).parts
    if any(part in skip_dirs for part in rel_parts):
        continue
    if any(part in forbidden_dirs for part in rel_parts):
        if p.is_file():
            found.append(p)
        continue
    if p.is_file() and any(p.match(pattern) for pattern in patterns):
        found.append(p)

if found:
    print('Potentially forbidden generated/archive files:')
    for p in found[:100]:
        print(' -', p.relative_to(ROOT))
    raise SystemExit(1)

print('PASS: no forbidden archive/cache/log files found in the repository.')
