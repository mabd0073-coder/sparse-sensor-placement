"""Validation of POD sensor placement on the RRF training and held-out test sweeps.

Takes rrf_train.npz and rrf_test.npz from cluster/load_rrf_sweeps.py and reports:

1. the number of POD modes for 90 % of the variation in the training set,
2. leave-one-out RMS on the training set,
3. RMS on each held-out test case, with modes and sensors from the training
   set only,
4. the mean-field baseline (training-average field, no sensors) for 2 and 3,

for greedy, greedy with a minimum spacing (--min-dist, metres), QR pivoting
and random layouts (mean over --seeds seeds) at 10, 20 and 24 sensors
(24 = one DPMS box). Placement and modes follow
experiments/placement_comparison.py (min_modes=1; QR on max(k, r) modes).

Usage:
    python experiments/rrf_validation.py data/rrf_train.npz data/rrf_test.npz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from placement_comparison import load_coords_m, qr_sensors, rms  # noqa: E402
from sensor_opt.core import compute_pod, greedy_sensor_placement, reconstruct  # noqa: E402


def place_all(U_r, U_big, r, ks, coords, min_dist, random_layouts):
    """{(method, k, seed): sensor indices} for every method and sensor count."""
    layouts = {}
    for k in ks:
        layouts[("greedy", k, -1)] = greedy_sensor_placement(U_r, k)
        if coords is not None:
            layouts[(f"greedy, min_dist {min_dist:g} m", k, -1)] = greedy_sensor_placement(
                U_r, k, coords=coords, min_dist=min_dist)
        layouts[("QR pivoting", k, -1)] = qr_sensors(U_big[:, :max(k, r)], k)
        for s, sel in enumerate(random_layouts[k]):
            layouts[("random", k, s)] = sel
    return layouts


def evaluate(X_train, X_eval, ks, a, coords, random_layouts):
    """Rows of (method, sensors, seed, case index, rms) for every column of X_eval."""
    U_r, mean, r = compute_pod(X_train, a.variance, min_modes=1)
    U_big, _, _ = compute_pod(X_train, a.variance, min_modes=max(ks))
    layouts = place_all(U_r, U_big, r, ks, coords, a.min_dist, random_layouts)
    rows = []
    for j in range(X_eval.shape[1]):
        truth = X_eval[:, j]
        for (method, k, seed), sel in layouts.items():
            rows.append(dict(method=method, sensors=k, seed=seed, case=j, modes=r,
                             rms=rms(reconstruct(U_r, mean, sel, truth[sel]), truth)))
        rows.append(dict(method="mean-field baseline", sensors=0, seed=-1, case=j, modes=r,
                         rms=rms(mean[:, 0], truth)))
    return rows


def summarise(df):
    order = list(dict.fromkeys(zip(df["method"], df["sensors"])))
    out = (df.groupby(["method", "sensors"], sort=False)["rms"]
           .agg(rms_mean="mean", rms_sd="std", rms_max="max").reset_index())
    out["key"] = list(zip(out["method"], out["sensors"]))
    return out.set_index("key").loc[order].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train", help="rrf_train.npz")
    p.add_argument("test", help="rrf_test.npz")
    p.add_argument("--key", default="Cp_matrix")
    p.add_argument("--sensors", type=int, nargs="+", default=[10, 20, 24])
    p.add_argument("--variance", type=float, default=0.90)
    p.add_argument("--min-dist", type=float, default=0.05)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default="results/rrf_validation")
    a = p.parse_args()

    tr, te = np.load(a.train), np.load(a.test)
    X, Y = tr[a.key], te[a.key]
    if X.shape[0] != Y.shape[0]:
        sys.exit(f"[error] train has {X.shape[0]} points, test has {Y.shape[0]}; they must share the target points")
    N, M = X.shape
    ks = a.sensors
    coords = load_coords_m(tr)
    if coords is None:
        print("NOTE: no x/y/z in the training npz; the min-distance method is skipped.")
    names = [str(n) for n in te["case_names"]] if "case_names" in te.files else [f"case {j}" for j in range(Y.shape[1])]
    print(f"Training: {a.train}, {N} points x {M} cases; test: {a.test}, {Y.shape[1]} cases")

    random_layouts = {k: [np.random.default_rng(s).choice(N, k, replace=False) for s in range(a.seeds)]
                      for k in ks}

    _, _, r_all = compute_pod(X, a.variance, min_modes=1)
    print(f"\nModes for {a.variance:.0%} of the variation (all {M} training cases): {r_all}")

    loo_rows = []
    for j in range(M):
        keep = np.arange(M) != j
        for row in evaluate(X[:, keep], X[:, [j]], ks, a, coords, random_layouts):
            loo_rows.append(dict(row, case=j))
    loo = pd.DataFrame(loo_rows)
    test = pd.DataFrame(evaluate(X, Y, ks, a, coords, random_layouts))
    test["case_name"] = [names[j] for j in test["case"]]

    loo_s = summarise(loo).rename(columns=lambda c: c.replace("rms", "loo_rms"))
    test_s = summarise(test).rename(columns=lambda c: c.replace("rms", "test_rms"))
    summary = loo_s.merge(test_s, on=["method", "sensors"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    summary.to_csv(a.out + "_summary.csv", index=False)
    loo.to_csv(a.out + "_loo.csv", index=False)
    test.to_csv(a.out + "_test_cases.csv", index=False)

    pd.set_option("display.width", 160)
    modes = loo.drop_duplicates("case")["modes"].value_counts().to_dict()
    print(f"Modes kept per leave-one-out fold: {modes}")
    print("\nSummary (random = pooled over seeds):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    per_case = test.groupby(["sensors", "case_name", "method"], sort=False)["rms"].mean().unstack("method")
    baseline = per_case.loc[0, "mean-field baseline"]
    for k in ks:
        block = per_case.loc[k].dropna(axis=1, how="all")
        block = block.rename(columns={"random": f"random (mean of {a.seeds})"})
        block["mean-field baseline"] = baseline
        print(f"\nRMS on each held-out test case, {k} sensors (r = {r_all} modes):")
        print(block.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved {a.out}_summary.csv, {a.out}_loo.csv and {a.out}_test_cases.csv")


if __name__ == "__main__":
    main()
