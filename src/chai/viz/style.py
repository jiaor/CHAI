"""One figure style for the whole project.

Extracted from ``scripts/plot_prior.py`` once a second consumer appeared
(``notebooks/data_audit.ipynb``).  Figures that share a palette and a set of
axis conventions read as one document; figures that each invent their own make
the reader re-learn the encoding on every page.

Call :func:`use` before plotting, and :func:`thermal_axes` on any axis whose
x is time before present and y is temperature.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["BLUES", "GRID", "INK", "INK_2", "MUTED", "SERIES", "SURFACE",
           "thermal_axes", "use"]

# Reference palette (docs: the dataviz skill's palette.md), used unchanged.
# Slots 1-4 in the palette's fixed order, assigned to entities and never
# cycled; slots 1-3 are the set validated on the all-pairs list, and the
# fourth (yellow) arrived with the Stage 11 figure's four modality classes.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

#: Sequential blue, 100 -> 700, for magnitude.  Zero fades into the surface.
BLUES = LinearSegmentedColormap.from_list("blues", [
    SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
    "#1c5cab", "#184f95", "#0d366b",
])

RCPARAMS = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "xtick.direction": "out", "ytick.direction": "out",
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
}


def use(dpi: int | None = None) -> None:
    """Apply the project rcParams.  ``dpi`` overrides the 200 used for files."""
    mpl.rcParams.update(RCPARAMS)
    if dpi is not None:
        mpl.rcParams["figure.dpi"] = dpi


def thermal_axes(ax: Axes, t_max: float, T_max: float) -> None:
    """The convention every thermochronology paper uses.

    Time before present runs right to left, so geological time reads forwards;
    temperature runs top to bottom, so shallow is up and the page is a cartoon
    of the crust.  Reading a t-T path drawn any other way takes a beat longer,
    every time.
    """
    ax.set_xlim(t_max, 0)
    ax.set_ylim(T_max, 0)
    ax.set_xlabel("time before present (Ma)")
    ax.set_ylabel("temperature (°C)")
    ax.grid(True, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def time_axis(ax: Axes, t_max: float, label: str) -> None:
    """Time before present right to left, with a free y -- the audit's workhorse.

    Same reading direction as :func:`thermal_axes`, for plots whose y is a
    diagnostic (a variance ratio, a sensitivity) rather than a temperature.
    """
    ax.set_xlim(t_max, 0)
    ax.set_xlabel("time before present (Ma)")
    ax.set_ylabel(label)
    ax.grid(True, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
