from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))

from style import configure_style
from figures import ALL

font=configure_style()
print(f"Using font: {font}")
fail=[]
for fn in ALL:
    try:
        print(f"[run] {fn.__name__}")
        fn()
        print(f"[ok ] {fn.__name__}")
    except Exception as e:
        fail.append((fn.__name__,repr(e)))
        print(f"[FAIL] {fn.__name__}: {e}")
if fail:
    print("\nSome figures failed:")
    for name,e in fail: print(" -",name,e)
    raise SystemExit(1)
print("\nAll figure scripts completed successfully.")
print("Outputs:", ROOT/"outputs_review")
