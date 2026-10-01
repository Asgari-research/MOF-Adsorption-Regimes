from pathlib import Path
import platform
import sys

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
for m in ["numpy", "pandas", "matplotlib", "PIL"]:
    try:
        mod = __import__(m)
        print(f"{m}: OK ({getattr(mod, '__version__', 'unknown')})")
    except Exception as e:
        print(f"{m}: MISSING ({e})")

font_candidates = [
    Path("/mnt/c/Windows/Fonts/arial.ttf"),
    Path("/mnt/c/Windows/Fonts/Arial.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
]
found = [str(p) for p in font_candidates if p.exists()]
print("Arial candidates found:", found if found else "none")
if not found:
    print("NOTE: plotting still works with DejaVu Sans, but it will not match the requested Arial typography.")
