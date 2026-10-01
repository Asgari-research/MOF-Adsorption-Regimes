from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from style import configure_style
from figures import figure1, figure2, figure3, figure4, si1, si2, si3, si4, si5, si6, si7, si8, toc

FIGURES = {
    "figure1": figure1, "fig1": figure1, "1": figure1,
    "figure2": figure2, "fig2": figure2, "2": figure2,
    "figure3": figure3, "fig3": figure3, "3": figure3,
    "figure4": figure4, "fig4": figure4, "4": figure4,
    "si1": si1, "s1": si1,
    "si2": si2, "s2": si2,
    "si3": si3, "s3": si3,
    "si4": si4, "s4": si4,
    "si5": si5, "s5": si5,
    "si6": si6, "s6": si6,
    "si7": si7, "s7": si7,
    "si8": si8, "s8": si8,
    "toc": toc,
}

if len(sys.argv) != 2 or sys.argv[1].lower() not in FIGURES:
    valid = "fig1 fig2 fig3 fig4 si1 si2 si3 si4 si5 si6 si7 si8 toc"
    print("Usage: python run_one_figure.py <figure>")
    print("Valid names:", valid)
    raise SystemExit(2)

key = sys.argv[1].lower()
font = configure_style()
print(f"Using font: {font}")
print(f"[run] {key}")
FIGURES[key]()
print(f"[ok ] {key}")
print("Outputs:", ROOT / "outputs_review")
