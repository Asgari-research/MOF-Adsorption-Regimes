from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []
warnings: list[str] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


# 1) Locked publication figures.
manifest = ROOT / 'integrity' / 'FINAL_FIGURE_MANIFEST.csv'
if not manifest.exists():
    errors.append(f'Missing final-figure manifest: {manifest}')
else:
    rows = list(csv.DictReader(manifest.open(encoding='utf-8', newline='')))
    for row in rows:
        name = row['filename']
        path = ROOT / 'figures' / 'final' / ('si' if name.startswith('Figure_S') else 'main') / name
        if not path.exists():
            errors.append(f'Missing locked figure: {path.relative_to(ROOT)}')
            continue
        actual = sha256(path)
        if actual != row['sha256']:
            errors.append(f'Locked figure hash mismatch: {path.relative_to(ROOT)}')

# 2) Figure-source data hashes.
data_manifest = ROOT / 'figure_generation' / 'DATA_SHA256.csv'
if not data_manifest.exists():
    errors.append('Missing figure_generation/DATA_SHA256.csv')
else:
    rows = list(csv.DictReader(data_manifest.open(encoding='utf-8', newline='')))
    for row in rows:
        path = ROOT / 'figure_generation' / row['relative_path']
        if not path.exists():
            errors.append(f'Missing figure source data: {path.relative_to(ROOT)}')
            continue
        if sha256(path) != row['sha256']:
            errors.append(f'Figure source-data hash mismatch: {path.relative_to(ROOT)}')

# 3) Frozen analysis source hashes.
frozen_manifest = ROOT / 'analysis' / 'frozen_original' / 'SHA256SUMS.txt'
if not frozen_manifest.exists():
    errors.append('Missing analysis/frozen_original/SHA256SUMS.txt')
else:
    for raw in frozen_manifest.read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if not raw:
            continue
        expected, rel = raw.split(None, 1)
        rel = rel.strip().lstrip('*')
        path = ROOT / 'analysis' / 'frozen_original' / rel
        if not path.exists():
            errors.append(f'Missing frozen-analysis file: {path.relative_to(ROOT)}')
            continue
        if sha256(path) != expected:
            errors.append(f'Frozen-analysis hash mismatch: {path.relative_to(ROOT)}')

# 4) Required repository documentation.
required = [
    'README.md',
    'docs/FINAL_FIGURE_LOCK.md',
    'docs/GITHUB_WSL2_RUNBOOK.md',
    'docs/DATA_PROVENANCE.md',
    'docs/SCIENTIFIC_REPRODUCIBILITY_NOTES.md',
    'docs/AUTHORSHIP_LICENSE_CITATION_PENDING.md',
    'analysis/audit_evidence/MISSING_INPUTS.md',
]
for rel in required:
    if not (ROOT / rel).exists():
        errors.append(f'Missing required repository file: {rel}')

# 5) Guard against accidental generated-review staging locations.
review_outputs = ROOT / 'figure_generation' / 'outputs_review'
if review_outputs.exists():
    warnings.append('figure_generation/outputs_review exists locally (expected after plotting; keep it untracked).')

# 6) Warn if archives are sitting inside repository root.
archives = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in {'.zip', '.7z'}]
if archives:
    warnings.append('Archive files found inside repository tree: ' + ', '.join(str(p.relative_to(ROOT)) for p in archives[:10]))

if warnings:
    print('WARNINGS:')
    for item in warnings:
        print(' -', item)

if errors:
    print('FAIL: repository verification found problems:')
    for item in errors:
        print(' -', item)
    raise SystemExit(1)

print('PASS: repository integrity checks succeeded.')
print(' - locked final figures: verified')
print(' - figure source data: verified')
print(' - frozen original analysis: verified')
print(' - required documentation: present')
