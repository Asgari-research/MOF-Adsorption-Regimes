from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []
warnings: list[str] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        errors.append(f'Missing manifest: {path.relative_to(ROOT)}')
        return []
    try:
        return list(csv.DictReader(path.open(encoding='utf-8', newline='')))
    except Exception as exc:
        errors.append(f'Could not parse manifest {path.relative_to(ROOT)}: {exc}')
        return []


# 1) Locked publication figures: exact set, byte count, and SHA-256.
expected_main = {f'Figure_{i}.pdf' for i in range(1, 5)}
expected_si = {f'Figure_S{i}.pdf' for i in range(1, 9)}
expected_figures = expected_main | expected_si
manifest = ROOT / 'integrity' / 'FINAL_FIGURE_MANIFEST.csv'
rows = manifest_rows(manifest)
manifest_names = [row.get('filename', '') for row in rows]
if len(rows) != 12:
    errors.append(f'Final-figure manifest must contain exactly 12 rows; found {len(rows)}')
if len(manifest_names) != len(set(manifest_names)):
    errors.append('Duplicate filenames found in final-figure manifest')
if set(manifest_names) != expected_figures:
    errors.append('Final-figure manifest filename set does not equal Figure_1-4 and Figure_S1-S8')

actual_final_files = {
    p.name for p in (ROOT / 'figures' / 'final' / 'main').glob('*.pdf')
} | {
    p.name for p in (ROOT / 'figures' / 'final' / 'si').glob('*.pdf')
}
if actual_final_files != expected_figures:
    missing = sorted(expected_figures - actual_final_files)
    extra = sorted(actual_final_files - expected_figures)
    errors.append(f'Final figure tree mismatch; missing={missing}, extra={extra}')

for row in rows:
    name = row.get('filename', '')
    path = ROOT / 'figures' / 'final' / ('si' if name.startswith('Figure_S') else 'main') / name
    if not path.exists():
        errors.append(f'Missing locked figure: {path.relative_to(ROOT)}')
        continue
    try:
        expected_bytes = int(row['bytes'])
        if path.stat().st_size != expected_bytes:
            errors.append(f'Locked figure byte-count mismatch: {path.relative_to(ROOT)}')
    except Exception:
        errors.append(f'Invalid byte count in final-figure manifest for {name}')
    if sha256(path) != row.get('sha256'):
        errors.append(f'Locked figure hash mismatch: {path.relative_to(ROOT)}')

# 2) Figure-source data: manifest must cover the complete curated data tree.
data_root = ROOT / 'figure_generation' / 'data'
data_manifest = ROOT / 'figure_generation' / 'DATA_SHA256.csv'
data_rows = manifest_rows(data_manifest)
manifest_data_paths = [row.get('relative_path', '') for row in data_rows]
if len(manifest_data_paths) != len(set(manifest_data_paths)):
    errors.append('Duplicate paths found in figure_generation/DATA_SHA256.csv')
actual_data_paths = sorted(str(p.relative_to(ROOT / 'figure_generation')).replace('\\', '/') for p in data_root.rglob('*') if p.is_file())
if set(manifest_data_paths) != set(actual_data_paths):
    missing = sorted(set(actual_data_paths) - set(manifest_data_paths))
    stale = sorted(set(manifest_data_paths) - set(actual_data_paths))
    errors.append(f'Figure-data manifest/tree mismatch; unmanifested={missing[:20]}, missing_files={stale[:20]}')
for row in data_rows:
    path = ROOT / 'figure_generation' / row.get('relative_path', '')
    if not path.exists():
        continue
    try:
        if path.stat().st_size != int(row['bytes']):
            errors.append(f'Figure source-data byte-count mismatch: {path.relative_to(ROOT)}')
    except Exception:
        errors.append(f'Invalid byte count in DATA_SHA256.csv for {row.get("relative_path", "")}')
    if sha256(path) != row.get('sha256'):
        errors.append(f'Figure source-data hash mismatch: {path.relative_to(ROOT)}')

# 3) Frozen original analysis hashes. README.md is the local wrapper; SHA256SUMS tracks original payloads.
frozen_root = ROOT / 'analysis' / 'frozen_original'
frozen_manifest = frozen_root / 'SHA256SUMS.txt'
frozen_entries: dict[str, str] = {}
if not frozen_manifest.exists():
    errors.append('Missing analysis/frozen_original/SHA256SUMS.txt')
else:
    for raw in frozen_manifest.read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            expected, rel = raw.split(None, 1)
        except ValueError:
            errors.append(f'Malformed frozen hash line: {raw}')
            continue
        rel = rel.strip().lstrip('*')
        if rel in frozen_entries:
            errors.append(f'Duplicate frozen hash entry: {rel}')
        frozen_entries[rel] = expected
        path = frozen_root / rel
        if not path.exists():
            errors.append(f'Missing frozen-analysis file: {path.relative_to(ROOT)}')
        elif sha256(path) != expected:
            errors.append(f'Frozen-analysis hash mismatch: {path.relative_to(ROOT)}')
expected_frozen_payload = {
    str(p.relative_to(frozen_root)).replace('\\', '/')
    for p in frozen_root.rglob('*') if p.is_file()
    and p.name not in {'README.md', 'SHA256SUMS.txt'}
}
if set(frozen_entries) != expected_frozen_payload:
    errors.append(
        'Frozen-analysis hash manifest/tree mismatch; '
        f'unmanifested={sorted(expected_frozen_payload - set(frozen_entries))}, '
        f'missing_files={sorted(set(frozen_entries) - expected_frozen_payload)}'
    )

# 4) Curated plotting source/config hashes.
portable_manifest = ROOT / 'integrity' / 'PORTABLE_PLOTTING_SHA256.txt'
if not portable_manifest.exists():
    errors.append('Missing integrity/PORTABLE_PLOTTING_SHA256.txt')
else:
    seen: set[str] = set()
    for raw in portable_manifest.read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            expected, rel = raw.split(None, 1)
        except ValueError:
            errors.append(f'Malformed portable plotting hash line: {raw}')
            continue
        rel = rel.strip().lstrip('*')
        if rel in seen:
            errors.append(f'Duplicate portable plotting hash entry: {rel}')
        seen.add(rel)
        path = ROOT / 'figure_generation' / rel
        if not path.exists():
            errors.append(f'Missing portable plotting file: {path.relative_to(ROOT)}')
        elif sha256(path) != expected:
            errors.append(f'Portable plotting hash mismatch: {path.relative_to(ROOT)}')

# 5) Required repository documentation and directories.
required = [
    'README.md',
    'docs/FINAL_FIGURE_LOCK.md',
    'docs/GITHUB_WSL2_RUNBOOK.md',
    'docs/GIT_COMMIT_PLAN.md',
    'docs/GIT_INCLUDE_EXCLUDE.md',
    'docs/DATA_PROVENANCE.md',
    'docs/FEATURE_AND_PREPROCESSING_MANIFEST.md',
    'docs/SCIENTIFIC_REPRODUCIBILITY_NOTES.md',
    'docs/AUTHORSHIP_LICENSE_CITATION_PENDING.md',
    'analysis/audit_evidence/MISSING_INPUTS.md',
    'analysis/audit_evidence/ISSUE_REGISTER.csv',
    'analysis/proposed_repairs/tests/test_identifier_matching.py',
]
for rel in required:
    if not (ROOT / rel).exists():
        errors.append(f'Missing required repository file: {rel}')

# 6) Python syntax only; do not import frozen scripts.
for path in sorted(ROOT.rglob('*.py')):
    if '.git' in path.parts:
        continue
    try:
        ast.parse(path.read_text(encoding='utf-8'))
    except Exception as exc:
        errors.append(f'Python syntax parse failed: {path.relative_to(ROOT)}: {exc}')

# 7) Guard against generated archives/caches and GitHub blob-size failures.
for path in ROOT.rglob('*'):
    if not path.is_file() or '.git' in path.parts:
        continue
    rel = path.relative_to(ROOT)
    if any(part in {'__pycache__', 'outputs_review', '.venv', 'venv', 'env'} for part in rel.parts):
        errors.append(f'Generated/local file present in repository tree: {rel}')
    if path.suffix.lower() in {'.zip', '.7z', '.pyc', '.log'}:
        errors.append(f'Archive/cache/log file present in repository tree: {rel}')
    if path.stat().st_size > 100 * 1024 * 1024:
        errors.append(f'File exceeds GitHub 100 MiB blob limit: {rel} ({path.stat().st_size} bytes)')

# 8) Lightweight secret/private-key pattern scan in readable text files.
secret_patterns = {
    'private key': re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY'),
    'GitHub token': re.compile(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}'),
    'AWS access key': re.compile(r'AKIA[0-9A-Z]{16}'),
}
text_suffixes = {'.md', '.py', '.csv', '.json', '.txt', '.yml', '.yaml', '.bat', ''}
for path in ROOT.rglob('*'):
    if not path.is_file() or '.git' in path.parts or path.suffix.lower() == '.pdf':
        continue
    if path.suffix.lower() not in text_suffixes and path.name != '.gitignore':
        continue
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        continue
    for label, pattern in secret_patterns.items():
        if pattern.search(text):
            errors.append(f'Potential {label} found in {path.relative_to(ROOT)}')

# 9) License/documentation consistency.
license_path = ROOT / 'LICENSE'
license_note = ROOT / 'docs' / 'AUTHORSHIP_LICENSE_CITATION_PENDING.md'
if license_path.exists() and license_note.exists():
    note = license_note.read_text(encoding='utf-8')
    if 'does **not** add a `LICENSE` file' in note or 'No software/data license was established' in note:
        errors.append('LICENSE exists but authorship/license note still states that no license file was added/established')
if not license_path.exists():
    warnings.append('No top-level LICENSE file is present; confirm intended licensing before public release.')

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
print(' - locked final figures: exact 12-file set, bytes, and SHA-256 verified')
print(' - figure source-data tree: complete manifest, bytes, and SHA-256 verified')
print(' - frozen original analysis: manifest coverage and SHA-256 verified')
print(' - portable plotting source/config: SHA-256 verified')
print(' - required documentation: present')
print(' - Python syntax: parsed without importing frozen scripts')
print(' - forbidden/generated files and >100 MiB blobs: none found')
print(' - lightweight secret/private-key scan: passed')
