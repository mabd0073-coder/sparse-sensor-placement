"""Drilling-tolerance study: how much does a mis-drilled pressure tap cost?

The model assumes each reading is taken at the intended point, but a tap drilled
delta mm away measures the pressure there. For each trial every sensor is moved
by delta mm in a random direction tangent to the local surface (plane fitted to
its 30 nearest points), snapped to the nearest CFD point, and the field is
rebuilt with the mode values U[intended] from the true readings X[actual].

Tests (fixed layout from results/feasible_layout.csv):
  B   POD trained on the straight cases, tested on the 22 cornering cases
  LOO POD refitted for each held-out case
Also reports, per sensor, the largest Cp change within 5 mm (local gradient),
which tells you where drilling accuracy matters most.

The layout was chosen with all cases, so the absolute numbers are slightly
optimistic; the increase over delta = 0 is what this study measures.

Usage:
    python experiments/drilling_tolerance.py data/side_diffuser_ensemble_highres_NN.npz \
        --params data/case_params.csv --layout results/feasible_layout.csv
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from cornering_test import pod  # noqa: E402

DELTAS_MM = (0, 1, 2, 3, 5, 7.5, 10, 15, 20)
TRIALS = 30


def tangents(coords, idx, tree, rng):
    """Random unit vectors in the tangent plane at each point idx (local PCA normal)."""
    out = []
    for i in idx:
        _, nb = tree.query(coords[i], k=30)
        P = coords[nb] - coords[nb].mean(0)
        n = np.linalg.svd(P, full_matrices=False)[2][-1]
        v = rng.normal(size=3)
        v -= v.dot(n) * n
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def rebuild_rms(U, mean, s_int, s_act, x):
    """RMS Cp error when readings come from s_act but the model assumes s_int."""
    a, *_ = np.linalg.lstsq(U[s_int], x[s_act] - mean[s_int, 0], rcond=None)
    return np.sqrt(np.mean((mean[:, 0] + U @ a - x) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--params", required=True)
    p.add_argument("--layout", default="results/feasible_layout.csv")
    p.add_argument("--variance", type=float, default=0.90)
    p.add_argument("--outdir", default="results")
    a = p.parse_args()
    t0 = time.time()

    z = np.load(a.ensemble)
    X = z["Cp_matrix"].astype(float)
    coords = np.column_stack([z["x"], z["y"], z["z"]]).astype(float)
    if np.ptp(coords, axis=0).max() > 20:
        coords /= 1000.0
    keep = coords[:, 1] <= 0
    X, coords = X[keep], coords[keep]
    prm = pd.read_csv(a.params)
    assert len(prm) == X.shape[1], "params rows must match ensemble columns"
    cr = prm["CR"].to_numpy(float)
    corner = (cr > 0) & (cr < 400)
    ride, cor = np.where(~corner)[0], np.where(corner)[0]
    L = pd.read_csv(a.layout)
    s = L.point_index.to_numpy()
    assert np.allclose(coords[s, 0] * 1000, L.x_mm, atol=0.1), "layout does not match this point set"

    tree = cKDTree(coords)
    spacing = tree.query(coords[::20], k=2)[0][:, 1] * 1000
    print(f"{len(coords)} points (y <= 0), nearest-neighbour aligned; median point spacing "
          f"{np.median(spacing):.1f} mm (5-95 %: {np.percentile(spacing, 5):.1f}-{np.percentile(spacing, 95):.1f})")

    # local sensitivity: largest |Cp change| within 5 mm, averaged over cases
    grad = []
    for j in s:
        nb = tree.query_ball_point(coords[j], 0.005)
        grad.append(np.abs(X[nb] - X[j]).max(axis=0).mean())
    L["max_dCp_within_5mm"] = grad

    rng = np.random.default_rng(3001)
    folds = [("B", list(ride), list(cor))] + [("LOO", [i for i in range(X.shape[1]) if i != j], [j])
                                             for j in range(X.shape[1])]
    rows = []
    for name, tr, te in folds:
        mean, U, r = pod(X[:, tr], a.variance)
        for d in DELTAS_MM:
            for t in range(1 if d == 0 else TRIALS):
                if d == 0:
                    act, off = s, np.zeros(len(s))
                else:
                    tgt = coords[s] + d / 1000 * tangents(coords, s, tree, rng)
                    act = tree.query(tgt)[1]
                    off = np.linalg.norm(coords[act] - coords[s], axis=1) * 1000
                for j in te:
                    rows.append(dict(test=name, delta_mm=d, trial=t, case=j, modes=r,
                                     realised_offset_mm=off.mean(),
                                     rms=rebuild_rms(U, mean, s, act, X[:, j])))
    df = pd.DataFrame(rows)
    df["test"] = df.test.where(df.test == "B", "LOO")
    os.makedirs(a.outdir, exist_ok=True)
    summ = (df.groupby(["test", "delta_mm"])
              .agg(mean_rms=("rms", "mean"), p95_rms=("rms", lambda v: np.percentile(v, 95)),
                   realised_offset_mm=("realised_offset_mm", "mean")).reset_index())
    base = summ[summ.delta_mm == 0].set_index("test").mean_rms
    summ["increase_pct"] = 100 * (summ.mean_rms / summ.test.map(base) - 1)
    summ.to_csv(f"{a.outdir}/drilling_tolerance.csv", index=False)
    L.to_csv(f"{a.outdir}/drilling_sensitivity_per_sensor.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMost position-sensitive sensors (largest Cp change within 5 mm):")
    print(L.sort_values("max_dCp_within_5mm", ascending=False)
           [["sensor", "x_mm", "y_mm", "z_mm", "max_dCp_within_5mm"]].head(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    for t, mk, lab in (("LOO", "o-", "leave-one-out, all cases"), ("B", "s--", "straight → cornering")):
        d = summ[summ.test == t]
        ax.plot(d.delta_mm, d.mean_rms, mk, ms=3.5, label=f"{lab} (mean)")
        ax.fill_between(d.delta_mm, d.mean_rms, d.p95_rms, alpha=0.15)
    ax.axvline(np.median(spacing), color="0.7", lw=0.8)
    ax.text(np.median(spacing) + 0.3, ax.get_ylim()[0] + 0.01, "mesh spacing", fontsize=6.5, color="0.4")
    ax.set_xlabel("drilling position error δ [mm]", fontsize=8)
    ax.set_ylabel("RMS error in Cp", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", title="shading: mean to 95th percentile",
              title_fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{a.outdir}/drilling_tolerance.png", dpi=300)
    print(f"\nSaved {a.outdir}/drilling_tolerance.csv/.png and drilling_sensitivity_per_sensor.csv; "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
