from pathlib import Path
import csv, hashlib
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'integrity'/'FINAL_FIGURE_MANIFEST.csv').open(encoding='utf-8')))
errors=[]
for r in rows:
    name=r['filename']
    p=(ROOT/'figures'/'final'/'main'/name) if not name.startswith('Figure_S') else (ROOT/'figures'/'final'/'si'/name)
    if not p.exists(): errors.append(f'MISSING: {p}'); continue
    if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']: errors.append(f'HASH MISMATCH: {name}')
if errors:
    print('\n'.join(errors)); raise SystemExit(1)
print(f'PASS: {len(rows)}/{len(rows)} locked final PDFs match the manifest.')
