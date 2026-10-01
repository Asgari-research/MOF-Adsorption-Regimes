from __future__ import annotations
import json
import platform
import sys

mods = {}
for name in ['numpy', 'pandas', 'matplotlib', 'PIL']:
    try:
        mod = __import__(name)
        mods[name] = getattr(mod, '__version__', 'unknown')
    except Exception as exc:
        mods[name] = f'MISSING: {exc}'

print(json.dumps({
    'python': sys.version,
    'platform': platform.platform(),
    'packages': mods,
}, indent=2))
