"""Sensor count and noise study on a nearest-neighbour ensemble (left side, y <= 0).

For k = 4..80 spaced-greedy sensors (min_dist 0.05 m), reports the RMS Cp error of
  (a) leave-one-out over all cases and
  (b) training on the straight-line cases and testing on the cornering cases,
with exact readings and with Gaussian reading noise of s.d. 0.03 in Cp (about the
team's +/-7 Pa DPMS uncertainty at 19.4 m/s). Greedy picks are nested, so one
search to the largest k gives every smaller layout.

Usage:
    python experiments/sensor_count_ensemble.py data/side_diffuser_ensemble_highres_NN.npz \
        --params data/case_params.csv
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cornering_test import pod, greedy  # noqa: E402

KS = (4, 6, 8, 10, 13, 16, 20, 24, 32, 40, 60, 80)
NOISE = 0.03
TRIALS = 10


def errors(X, tr, te, var, coords, md, rng):
    mean, U, r = pod(X[:, tr], var)
    sel = greedy(U, max(KS), coords, md)
    out = []
    for k in KS:
        s = sel[:k]
        for j in te:
            a, *_ = np.linalg.lstsq(U[s], X[s, j] - mean[s, 0], rcond=None)
            e0 = np.sqrt(np.mean((mean[:, 0] + U @ a - X[:, j]) ** 2))
            en = []
            for _ in range(TRIALS):
                y = X[s, j] + rng.normal(0, NOISE, len(s))
                a, *_ = np.linalg.lstsq(U[s], y - mean[s, 0], rcond=None)
                en.append(np.sqrt(np.mean((mean[:, 0] + U @ a - X[:, j]) ** 2)))
            base = np.sqrt(np.mean((mean[:, 0] - X[:, j]) ** 2))
            out.append(dict(k=k, case=j, modes=r, rms=e0, rms_noisy=np.mean(en), baseline=base))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--params", required=True)
    p.add_argument("--variance", type=float, default=0.90)
    p.add_argument("--min-dist", type=float, default=0.05)
    p.add_argument("--outdir", default="results")
    a = p.parse_args()

    z = np.load(a.ensemble)
    X = z["Cp_matrix"].astype(float)
    coords = np.column_stack([z["x"], z["y"], z["z"]]).astype(float)
    keep = coords[:, 1] <= 0
    X, coords = X[keep], coords[keep]
    prm = pd.read_csv(a.params)
    assert len(prm) == X.shape[1], "params rows must match ensemble columns"
    corner = (prm["CR"].to_numpy(float) > 0) & (prm["CR"].to_numpy(float) < 400)
    ride, cor = np.where(~corner)[0], np.where(corner)[0]
    M = X.shape[1]
    rng = np.random.default_rng(3001)
    print(f"{X.shape[0]} points (y <= 0), {len(ride)} straight + {len(cor)} cornering cases; "
          "nearest-neighbour aligned")

    rows = []
    for j in range(M):
        for d in errors(X, [i for i in range(M) if i != j], [j], a.variance, coords, a.min_dist, rng):
            rows.append(dict(test="a_LOO_all", **d))
    for d in errors(X, list(ride), list(cor), a.variance, coords, a.min_dist, rng):
        rows.append(dict(test="b_train_ride_test_corner", **d))
    df = pd.DataFrame(rows)
    tag = os.path.splitext(os.path.basename(a.ensemble))[0]
    os.makedirs(a.outdir, exist_ok=True)
    df.to_csv(f"{a.outdir}/sensor_count_{tag}_folds.csv", index=False)
    s = df.groupby(["test", "k"])[["rms", "rms_noisy", "baseline", "modes"]].mean().reset_index()
    s.to_csv(f"{a.outdir}/sensor_count_{tag}.csv", index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot(s, NOISE, f"{a.outdir}/sensor_count_{tag}.png")
    print(f"Saved {a.outdir}/sensor_count_{tag}.csv/.png")


def plot(s, noise, out):
    """Two-panel figure from the summary table (also usable on a saved CSV)."""
    from matplotlib.ticker import NullFormatter
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, t, ttl in zip(axs, ("a_LOO_all", "b_train_ride_test_corner"),
                          ("(a) leave-one-out, all 122 cases", "(b) train on straight, test on cornering")):
        d = s[s.test == t]
        ax.plot(d.k, d.rms, "o-", ms=3.5, label="exact readings")
        ax.plot(d.k, d.rms_noisy, "s--", ms=2.5, label=f"noise s.d. {noise} in Cp")
        ax.axhline(d.baseline.iloc[0], color="0.4", lw=1, ls=":", label="no sensors (average field)")
        r = d.modes.iloc[0]
        ax.axvline(r, color="0.75", lw=0.8)
        ax.text(r * 1.04, 0.03, f"k = r = {r:.0f} modes", fontsize=7, color="0.35")
        ax.set_xscale("log")
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.set_xticklabels(["4", "8", "16", "32", "64"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("number of sensors k (spaced greedy)", fontsize=8)
        ax.set_title(ttl, fontsize=8)
        ax.tick_params(labelsize=7.5)
    axs[0].set_ylabel("mean RMS error in Cp", fontsize=8)
    axs[0].set_ylim(0, 0.6)
    axs[0].legend(fontsize=7, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)


if __name__ == "__main__":
    main()
