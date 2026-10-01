from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager

COL = {
    "ink": "#1F2937",
    "muted": "#667085",
    "grid": "#DCE3E8",
    "blue": "#356F9F",
    "sky": "#8DBDD8",
    "teal": "#2A9D8F",
    "orange": "#E59B2F",
    "coral": "#D95D54",
    "red": "#B6404D",
    "purple": "#7566A8",
    "green": "#5D9C70",
    "navy": "#315B7D",
    "ocean": "#3F7F8C",
    "amber": "#D49A3A",
    "salmon": "#D46A5D",
    "grey": "#A7B0B8",
    "pale_blue": "#EFF5F9",
    "pale_teal": "#EEF8F6",
    "pale_coral": "#FFF2F0",
    "pale_grey": "#F6F7F8",
}
CYCLE = [COL["blue"], COL["teal"], COL["orange"], COL["purple"], COL["coral"], COL["green"]]


def configure_style():
    """Prefer Arial on Windows, with an explicit fallback warning."""
    arial_candidates = [
        # Windows-native execution
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
        # Ubuntu/WSL2 with the Windows C: drive mounted at /mnt/c
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
        Path("/mnt/c/Windows/Fonts/Arial.ttf"),
        # Linux font packages, when present
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
    ]
    for path in arial_candidates:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:
                pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in names else "DejaVu Sans"
    if font != "Arial":
        print("[font warning] Arial was not found. Falling back to DejaVu Sans for this run.")
        print("[font warning] On WSL2, Arial is normally available at /mnt/c/Windows/Fonts/arial.ttf when the C: drive is mounted.")

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": font,
        "font.size": 12.0,
        "axes.titlesize": 13.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 12.4,
        "xtick.labelsize": 10.9,
        "ytick.labelsize": 10.9,
        "legend.fontsize": 10.1,
        "axes.edgecolor": COL["ink"],
        "axes.linewidth": 0.9,
        "grid.color": COL["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.65,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
    })
    return font


def clean_axes(ax, grid=True):
    if grid:
        ax.grid(True, axis="both", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=COL["ink"])


def panel_label(ax, label, x=-0.10, y=1.05):
    """Large unboxed capital panel label used consistently across all figures."""
    kw=dict(
        fontsize=16.5, fontweight="bold", ha="left", va="bottom", color=COL["ink"],
        clip_on=False, zorder=50,
    )
    label=str(label).upper()
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, label, transform=ax.transAxes, **kw)
    else:
        ax.text(x, y, label, transform=ax.transAxes, **kw)
