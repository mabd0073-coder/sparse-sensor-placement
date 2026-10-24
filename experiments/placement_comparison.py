"""Leave-one-out comparison of sensor placement methods.

For every case of a snapshot ensemble, the POD basis is rebuilt from the other
cases, sensors are placed with each method, and the held-out case is
reconstructed from its readings at those sensors. Methods:

- greedy D-optimal (``sensor_opt.greedy_sensor_placement``)
- greedy with a minimum spacing between sensors (``--min-dists``, metres)
- QR column pivoting on the first ``max(k, r)`` POD modes (standard
  oversampling when there are more sensors than retained modes)
- random layouts, one fixed layout per seed, averaged over ``--seeds`` seeds
- the mean-field baseline (training-average field, no sensors)

Modes follow the 90 % variance rule with no lower bound (``min_modes=1``), as in
experiments/validation_extras.py, so the plain greedy result matches that script.

Usage:
    python experiments/placement_comparison.py data/side_diffuser_ensemble_dedup.npz
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensor_opt.core import compute_pod, greedy_sensor_placement, reconstruct  # noqa: E402


def load_coords_m(npz):
    """Point coordinates in metres, or None if the file has none."""
    if not all(k in npz.files for k in ("x", "y", "z")):
        return None
    coords = np.column_stack([npz["x"], npz["y"], npz["z"]]).astype(float)
    if np.abs(coords).max() > 10.0:  # stored in mm
        print("Coordinates look like millimetres; converting to metres.")
        coords /= 1000.0
    return coords


def qr_sensors(U_k, k):
    """First k QR pivots of U_k^T."""
    _, piv = qr(U_k.T, mode="r", pivoting=True)
    return piv[:k]


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def min_spacing(coords, sel):
    p = coords[sel]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return float(d[np.triu_indices(len(sel), k=1)].min())


def plot(summary, out_png, title, subtitle):
    ink, ink2, bar, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    s = summary[summary["method"] != "mean-field baseline"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(s) + 1.6), facecolor=surface)
    ax.set_facecolor(surface)
    y = np.arange(len(s))
    ax.barh(y, s["loo_rms_mean"], height=0.55, color=bar, xerr=s["loo_rms_sd"],
            error_kw=dict(ecolor=ink2, elinewidth=1, capsize=3))
    for yi, v in zip(y, s["loo_rms_mean"]):
        ax.text(v, yi + 0.33, f" {v:.4f}", va="bottom", ha="left", fontsize=8, color=ink2)
    ax.set_yticks(y, s["method"], color=ink, fontsize=9)
    ax.set_xlabel("Leave-one-out RMS error in Cp (bar: mean, whisker: s.d. over held-out cases)",
                  color=ink2, fontsize=8)
    ax.tick_params(axis="x", colors=ink2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.set_xlim(0, None)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.text(0.02, 0.97, title, ha="left", va="top", fontsize=11, color=ink)
    fig.text(0.02, 0.915, subtitle, ha="left", va="top", fontsize=8, color=ink2, linespacing=1.4)
    fig.savefig(out_png, dpi=200, facecolor=surface)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble", help=".npz with Cp_matrix (points x cases) and x, y, z")
    p.add_argument("--key", default="Cp_matrix")
    p.add_argument("--sensors", type=int, default=10)
    p.add_argument("--variance", type=float, default=0.90)
    p.add_argument("--min-dists", type=float, nargs="*", default=[0.02, 0.05, 0.10])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default="results/placement_comparison")
    a = p.parse_args()

    npz = np.load(a.ensemble)
    X = npz[a.key]
    N, M = X.shape
    k = a.sensors
    print(f"Ensemble {a.ensemble}: {N} points x {M} cases; {k} sensors")
    coords = load_coords_m(npz)
    min_dists = a.min_dists
    if coords is None:
        print("NOTE: the npz has no x/y/z coordinates; min-distance methods are skipped.")
        min_dists = []

    layouts_random = [np.random.default_rng(s).choice(N, k, replace=False) for s in range(a.seeds)]

    rows = []
    for j in range(M):
        train = np.arange(M) != j
        U_r, mean, r = compute_pod(X[:, train], a.variance, min_modes=1)
        U_k, _, _ = compute_pod(X[:, train], a.variance, min_modes=k)
        truth = X[:, j]

        layouts = {"greedy": greedy_sensor_placement(U_r, k)}
        for d in min_dists:
            layouts[f"greedy, min_dist {d:g} m"] = greedy_sensor_placement(U_r, k, coords=coords, min_dist=d)
        layouts["QR pivoting"] = qr_sensors(U_k, k)
        for name, sel in layouts.items():
            rows.append(dict(method=name, seed=-1, fold=j, modes=r,
                             rms=rms(reconstruct(U_r, mean, sel, truth[sel]), truth)))
        for s, sel in enumerate(layouts_random):
            rows.append(dict(method="random", seed=s, fold=j, modes=r,
                             rms=rms(reconstruct(U_r, mean, sel, truth[sel]), truth)))
        rows.append(dict(method="mean-field baseline", seed=-1, fold=j, modes=r, rms=rms(mean[:, 0], truth)))
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{M} folds done")

    folds = pd.DataFrame(rows)
    order = list(dict.fromkeys(folds["method"]))
    summary = (folds.groupby("method", sort=False)["rms"]
               .agg(loo_rms_mean="mean", loo_rms_sd="std", loo_rms_max="max").reindex(order).reset_index())
    per_seed = folds[folds["method"] == "random"].groupby("seed")["rms"].mean()
    summary["random_seed_sd"] = np.where(summary["method"] == "random", per_seed.std(), np.nan)

    # Spacing of the layout each method picks when trained on all cases.
    if coords is not None:
        U_all, _, _ = compute_pod(X, a.variance, min_modes=1)
        U_all_k, _, _ = compute_pod(X, a.variance, min_modes=k)
        full = {"greedy": greedy_sensor_placement(U_all, k), "QR pivoting": qr_sensors(U_all_k, k)}
        for d in min_dists:
            full[f"greedy, min_dist {d:g} m"] = greedy_sensor_placement(U_all, k, coords=coords, min_dist=d)
        summary["min_spacing_m"] = summary["method"].map({m: min_spacing(coords, s) for m, s in full.items()})
        summary.loc[summary["method"] == "random", "min_spacing_m"] = np.mean(
            [min_spacing(coords, s) for s in layouts_random])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    summary.to_csv(a.out + ".csv", index=False)
    folds.to_csv(a.out + "_folds.csv", index=False)

    modes = folds.drop_duplicates("fold")["modes"].value_counts().to_dict()
    pd.set_option("display.width", 140)
    print(f"\nModes kept per fold: {modes}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    base = summary.loc[summary["method"] == "mean-field baseline", "loo_rms_mean"].item()
    plot(summary, a.out + ".png",
         f"Sensor placement, leave-one-out over {M} cases ({k} sensors)",
         f"{os.path.basename(a.ensemble)}; random = mean of {a.seeds} seeds\n"
         f"Mean-field baseline (no sensors) = {base:.4f}, not drawn")
    print(f"\nSaved {a.out}.csv, {a.out}_folds.csv and {a.out}.png")


if __name__ == "__main__":
    main()
