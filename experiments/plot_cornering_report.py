"""Report figure comparing the cornering test on the two nearest-neighbour ensembles.

Reads the CSVs written by experiments/cornering_test.py and plots mean RMS error
(bars) and worst case (ticks) at 24 sensors. Usage:
    python experiments/plot_cornering_report.py results/cornering_test_side_diffuser_ensemble_highres_NN.csv \
        results/cornering_test_side_diffuser_ensemble.csv -o results/fig_cornering.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("whole_car_csv")
p.add_argument("left_only_csv")
p.add_argument("-o", "--out", default="results/fig_cornering.png")
p.add_argument("--sensors", type=int, default=24)
p.add_argument("--column", action="store_true", help="stack panels for one journal column")
a = p.parse_args()

sets = [(pd.read_csv(a.whole_car_csv), "Ensemble B1 (whole-car mesh, left half)", "#2a6fdb"),
        (pd.read_csv(a.left_only_csv), "Ensemble B2 (left-side mesh)", "#e8833a")]
methods = ["greedy", "greedy_md", "qr", "random", "mean_field_baseline"]
labels = ["greedy", "greedy\nspaced", "QR", "random", "no\nsensors"]
tests = [("a_LOO_all", "(a) leave-one-out, all 122"),
         ("b_train_ride_test_corner", "(b) train straight, test 22 cornering"),
         ("c_LOO_corner_only", "(c) leave-one-out, 22 cornering")]
YMAX = 0.9
if a.column:
    fig, axs = plt.subplots(3, 1, figsize=(3.45, 4.5), sharex=True)
else:
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
for ax, (t, ttl) in zip(axs, tests):
    x = np.arange(len(methods))
    modes = []
    for i, (df, name, c) in enumerate(sets):
        d = df[(df.test == t) & (df.sensors == a.sensors)].set_index("method").loc[methods]
        xs = x + (i - 0.5) * 0.38
        ax.bar(xs, np.minimum(d.mean_rms, YMAX), 0.38, color=c, label=name)
        ax.plot(xs, np.minimum(d.worst_rms, YMAX * 0.995), "_", color="k", ms=6, mew=1.2)
        for xx, v in zip(xs, d.mean_rms):
            if v > YMAX:
                ax.text(xx, YMAX * 0.9, f"{v:.1f}", ha="center", fontsize=6, rotation=90, color="w")
        modes.append(int(d["modes_median"].iloc[0]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_title(f"{ttl}\nmodes kept: B1 {modes[0]}, B2 {modes[1]}", fontsize=7.5)
    ax.set_ylim(0, YMAX)
    ax.tick_params(labelsize=7)
if a.column:
    fig.supylabel(f"RMS error in Cp, {a.sensors} sensors", fontsize=7.5)
else:
    axs[0].set_ylabel(f"RMS error in Cp, {a.sensors} sensors", fontsize=7.5)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=1 if a.column else 2, fontsize=6.5 if a.column else 7, frameon=False)
fig.tight_layout(rect=(0, 0.05 if a.column else 0.08, 1, 1))
fig.savefig(a.out, dpi=220)
print("saved", a.out)
