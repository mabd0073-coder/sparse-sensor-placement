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


if __name__ == "__main__":
    main()
