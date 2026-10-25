"""Plot how the new space-filling sweep designs cover the parameter space.

Produces
  design_simple.png : two panels for the report body (old sweeps vs new designs)
  design_full.png   : six panels incl. roll and yaw (report appendix)

Usage:  python sweeps/plot_design_coverage.py --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
B1, B2, TE, OLD, INK = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b", "#1a1a19"


def style(ax):
    ax.grid(color="#e8e7e1", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors="#55554f")


def sc(ax, d, x, y, col, m, lab, s=24):
    ax.scatter(d[x], d[y], c=col, marker=m, s=s, label=lab, edgecolors="white", linewidths=0.6, zorder=3)


def simple(team, c1, tc, s1, ts, G, out, column):
    if column:
        fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.6), gridspec_kw=dict(hspace=1.15))
        fs, tfs = 6.5, 7.5
    else:
        fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw=dict(wspace=0.28))
        fs, tfs = 7.2, 8.8
    a = ax[0]
    a.scatter(team.cr, np.full(len(team), 31.5), c=OLD, s=22, label="old: sweep 50.34 (29 cases)", zorder=2)
    sc(a, c1, "cr", "frh", B1, "o", "new: cornering batch 1 (32)")
    sc(a, tc, "cr", "frh", TE, "^", "new: held-out test (8)", 40)
    a.set(xlabel="corner radius [m]", ylabel="front ride height [mm]", ylim=(8, 62))
    a.set_title("(a) Cornering: the old sweep is one line;\nthe new cases fill the range", fontsize=tfs, loc="left", color=INK)
    a.legend(frameon=False, fontsize=fs, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.3 if column else -0.2))
    style(a)
    a = ax[1]
    a.scatter(G[:, 0], G[:, 1], c=OLD, marker="+", s=26, label="old: sweep 50.28 grid (100)", zorder=2)
    sc(a, s1, "frh", "rrh", B1, "o", "new: straight-line batch 1 (32)")
    sc(a, ts, "frh", "rrh", TE, "^", "new: held-out test (8)", 40)
    a.set(xlabel="front ride height [mm]", ylabel="rear ride height [mm]", xlim=(11, 59), ylim=(11, 59))
    a.set_title("(b) Straight line: the new cases also\nvary roll and yaw (not shown)", fontsize=tfs, loc="left", color=INK)
    a.legend(frameon=False, fontsize=fs, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.3 if column else -0.2))
    style(a)
    fig.savefig(out / ("design_simple_col.png" if column else "design_simple.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", default="results/figures")
    out = Path(pa.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})

    team = pd.read_csv(HERE / "steer2moreres_50.34.csv")
    rd = lambda f: pd.read_csv(HERE / "new" / f)
    c1, c2 = rd("cornering_rrf_batch1.csv"), rd("cornering_rrf_batch2.csv")
    s1, s2 = rd("straight_rrf_batch1.csv"), rd("straight_rrf_batch2.csv")
    t = rd("heldout_rrf_test.csv")
    tc, ts = t[t.cr < 100], t[t.cr > 100]
    g = np.linspace(15, 55, 10)
    G = np.array([(a, b) for a in g for b in g])

    # ---- simple two-panel version (wide, and stacked for one journal column) ---
    for column in (False, True):
        simple(team, c1, tc, s1, ts, G, out, column)

    # ---- full six-panel version (appendix) -------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(8.8, 5.6))
    a = ax[0, 0]
    a.plot(team.cr, team.s, color=OLD, lw=2, label="50.34 (existing line)")
    sc(a, c1, "cr", "s", B1, "o", "batch 1")
    sc(a, c2, "cr", "s", B2, "s", "batch 2")
    sc(a, tc, "cr", "s", TE, "^", "held-out test", 40)
    a.set(xlabel="corner radius cr [m]", ylabel="steer s [deg]", title="Cornering: steer follows the team curve")
    a = ax[0, 1]
    a.scatter(team.cr, np.full(len(team), 31.5), c=OLD, s=18, label="50.34 (all at 31.5)")
    sc(a, c1, "cr", "frh", B1, "o", "batch 1")
    sc(a, c2, "cr", "frh", B2, "s", "batch 2")
    sc(a, tc, "cr", "frh", TE, "^", "held-out test", 40)
    a.set(xlabel="corner radius cr [m]", ylabel="front ride height [mm]", title="Cornering: radius vs ride height")
    a = ax[0, 2]
    sc(a, c1, "r", "y", B1, "o", "batch 1")
    sc(a, c2, "r", "y", B2, "s", "batch 2")
    sc(a, tc, "r", "y", TE, "^", "held-out test", 40)
    a.scatter([0], [0], c=OLD, s=40, label="50.34 (all at 0, 0)")
    a.set(xlabel="roll r [deg]", ylabel="yaw y [deg]", title="Cornering: roll vs yaw")
    a = ax[1, 0]
    a.scatter(G[:, 0], G[:, 1], c=OLD, s=14, marker="+", label="50.28 (grid)")
    sc(a, s1, "frh", "rrh", B1, "o", "batch 1")
    sc(a, s2, "frh", "rrh", B2, "s", "batch 2")
    sc(a, ts, "frh", "rrh", TE, "^", "held-out test", 40)
    a.set(xlabel="front ride height [mm]", ylabel="rear ride height [mm]", title="Straight: ride heights")
    a = ax[1, 1]
    xx = np.linspace(15, 55, 100)
    lim = 3 * np.clip((xx - 15) / 16.5, 0, 1)
    a.fill_between(xx, -lim, lim, color="#f0efec", label="allowed roll")
    for d, col, m, lab, s in [(s1, B1, "o", "batch 1", 26), (s2, B2, "s", "batch 2", 26), (ts, TE, "^", "held-out test", 40)]:
        a.scatter(np.minimum(d.frh, d.rrh), d.r, c=col, marker=m, s=s, label=lab, edgecolors="white", linewidths=0.6)
    a.set(xlabel="lower of FRH / RRH [mm]", ylabel="roll r [deg]", title="Roll limited at low ride height")
    a = ax[1, 2]
    sc(a, s1, "r", "y", B1, "o", "batch 1")
    sc(a, s2, "r", "y", B2, "s", "batch 2")
    sc(a, ts, "r", "y", TE, "^", "held-out test", 40)
    a.set(xlabel="roll r [deg]", ylabel="yaw y [deg]", title="Straight: roll vs yaw")
    for a in ax.flat:
        a.legend(fontsize=5.8, frameon=True, framealpha=0.85, loc="best", markerscale=0.8)
        a.title.set_fontsize(8)
        a.xaxis.label.set_fontsize(7.5)
        a.yaxis.label.set_fontsize(7.5)
        style(a)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "design_full.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
