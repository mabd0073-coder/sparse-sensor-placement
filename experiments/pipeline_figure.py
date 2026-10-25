"""Workflow diagram for the report, drawn for one journal column (vertical flow).

Usage: python experiments/pipeline_figure.py --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, BLUE_BG = "#2a6fdb", "#edf3fc"
ORANGE, ORANGE_BG = "#e8642c", "#fdf0e9"
INK = "#1f1f1c"

STEPS = [
    ("CFD cases", "Fluent on M3; one surface file per case (Cp at ~145,000 points)"),
    ("Map onto one point set", "each case's mesh mapped onto one reference mesh; left side"),
    ("Snapshot matrix X", "one column per case (never averaged); subtract the mean field"),
    ("POD by SVD", "keep the r modes that carry 90 % of the variation"),
    ("Choose k sensors", "greedy D-optimal; ≥ 0.05 m apart; away from edges and CAD zones"),
]
CHECK = ("Validate", "hide a case; rebuild it from k readings, x̂ = x̄ + U_r a;\n"
         "RMS error against its own CFD field and the no-sensor baseline")


def box(ax, y, h, title, body, edge, bg):
    ax.add_patch(FancyBboxPatch((0.04, y), 0.92, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=bg, ec=edge, lw=1.2))
    ax.text(0.08, y + h - 0.022, title, fontsize=7, weight="bold", color=INK, va="top")
    ax.text(0.08, y + h - 0.06, body, fontsize=6.1, color=INK, va="top", wrap=True)


SHORT = [
    ("CFD cases", "Fluent on M3;\none surface file\nper case (Cp at\n~145,000 points)"),
    ("Map onto one\npoint set", "each case's mesh\nmapped onto one\nreference mesh;\nleft side"),
    ("Snapshot\nmatrix X", "one column per\ncase (never\naveraged); subtract\nthe mean field"),
    ("POD by SVD", "keep the r modes\nthat carry 90 %\nof the variation"),
    ("Choose k\nsensors", "greedy D-optimal;\n≥ 0.05 m apart;\naway from edges\nand CAD zones"),
    ("Validate", "hide a case; rebuild\nfrom k readings,\nx̂ = x̄ + U_r a; RMS\nvs CFD and baseline"),
]


def horizontal(out):
    """Same workflow as one row across the page (two-column report, full width)."""
    fig = plt.figure(figsize=(7.1, 1.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    n, gap = len(SHORT), 0.022
    w = (0.995 - 0.005 - gap * (n - 1)) / n
    for i, (t, b) in enumerate(SHORT):
        x = 0.005 + i * (w + gap)
        last = i == n - 1
        ax.add_patch(FancyBboxPatch((x, 0.04), w, 0.92, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    fc=ORANGE_BG if last else BLUE_BG, ec=ORANGE if last else BLUE, lw=1.1))
        ax.text(x + 0.008, 0.9, t, fontsize=7, weight="bold", color=INK, va="top", linespacing=1.05)
        ax.text(x + 0.008, 0.9 - (0.33 if "\n" in t else 0.2), b, fontsize=5.9, color=INK, va="top",
                linespacing=1.15)
        if not last:
            ax.annotate("", xy=(x + w + gap - 0.002, 0.5), xytext=(x + w + 0.002, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    fig.savefig(out / "fig_pipeline_row.png", dpi=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures")
    out = Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(3.45, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    h, gap = 0.115, 0.04
    y = 0.985 - h
    tops = []
    for t, b in STEPS:
        box(ax, y, h, t, b, BLUE, BLUE_BG)
        tops.append(y)
        y -= h + gap
    hv = 0.15
    box(ax, y - hv + h, hv, *CHECK, ORANGE, ORANGE_BG)
    for y0 in tops:
        ax.annotate("", xy=(0.5, y0 - gap + 0.004), xytext=(0.5, y0 - 0.004),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    for txt in ax.texts:                        # wrap long lines inside the boxes
        txt._get_wrap_line_width = lambda: 0.84 * fig.bbox.width
    fig.savefig(out / "fig1_pipeline_col.png", dpi=300)
    horizontal(out)


if __name__ == "__main__":
    main()
