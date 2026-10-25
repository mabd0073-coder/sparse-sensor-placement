"""Does truncating the modes to the sensor count (or a ridge prior) help when k < r?

With k sensors and r modes, least squares on U_s (k x r) is underdetermined when
k < r and returns the minimum-norm coefficients. Three rebuilds are compared,
all with spaced greedy sensors (min_dist 0.05 m) chosen on the training cases:

  full   r from the 90 % variance rule, plain least squares (the report's method)
  trunc  only the first min(r, k) modes, both for placement and the rebuild
  ridge  all r modes, with a Gaussian prior on the coefficients from the training
         singular values: a = (U_s'U_s + s^2 L^-1)^-1 U_s'y, L = diag(lambda/(M-1)),
         s = 0.03 (the sensor-noise level used in the report). Nothing is tuned.

Tests: (a) leave-one-out over all cases, (b) train straight, test cornering.

Usage:
    python experiments/truncation_test.py data/side_diffuser_ensemble_highres_NN.npz \
        --params data/case_params.csv --tag B1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cornering_test import greedy  # noqa: E402

MIN_DIST, SIGMA, VAR = 0.05, 0.03, 0.90


def pod_full(Xtr):
    mean = Xtr.mean(axis=1, keepdims=True)
    Xc = Xtr - mean
    w, V = np.linalg.eigh(Xc.T @ Xc)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    r = int(np.searchsorted(np.cumsum(w) / w.sum(), VAR) + 1)
    r = min(r, int((w > 1e-12 * w[0]).sum()))
    U = (Xc @ V[:, :r]) / np.sqrt(w[:r])
    return mean[:, 0], U, w[:r] / (Xtr.shape[1] - 1), r


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fold(X, coords, tr, te, k):
    mean, U, lam, r = pod_full(X[:, tr])
    out = {"modes": r}
    s = greedy(U, k, coords, MIN_DIST)
    Us = U[s]
    rk = min(r, k)
    st = greedy(U[:, :rk], k, coords, MIN_DIST)
    P = SIGMA ** 2 * np.diag(1.0 / lam)
    for name in ("full", "trunc", "ridge", "baseline"):
        out[name] = []
    for j in te:
        x = X[:, j]
        y = x[s] - mean[s]
        a = np.linalg.lstsq(Us, y, rcond=None)[0]
        out["full"].append(rms(mean + U @ a, x))
        at = np.linalg.lstsq(U[st, :rk], x[st] - mean[st], rcond=None)[0]
        out["trunc"].append(rms(mean + U[:, :rk] @ at, x))
        ar = np.linalg.solve(Us.T @ Us + P, Us.T @ y)
        out["ridge"].append(rms(mean + U @ ar, x))
        out["baseline"].append(rms(mean, x))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--params", required=True)
    p.add_argument("--tag", default="B1")
    p.add_argument("--sensors", type=int, nargs="+", default=[10, 20])
    p.add_argument("--outdir", default="results/k20")
    a = p.parse_args()

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
    ride, cor = list(np.where(~corner)[0]), list(np.where(corner)[0])
    M = X.shape[1]
    print(f"{a.tag}: {X.shape[0]} points x {M} cases ({len(cor)} cornering)")

    rows = []
    for k in a.sensors:
        o = fold(X, coords, ride, cor, k)
        rows.append(dict(test="straight->cornering", k=k, modes_median=o["modes"],
                         **{m: np.mean(o[m]) for m in ("full", "trunc", "ridge", "baseline")},
                         **{f"{m}_worst": np.max(o[m]) for m in ("full", "trunc", "ridge")}))
        acc = {m: [] for m in ("full", "trunc", "ridge", "baseline")}
        modes = []
        for j in range(M):
            o = fold(X, coords, [i for i in range(M) if i != j], [j], k)
            modes.append(o["modes"])
            for m in acc:
                acc[m] += o[m]
        rows.append(dict(test="leave-one-out", k=k, modes_median=int(np.median(modes)),
                         **{m: np.mean(acc[m]) for m in acc},
                         **{f"{m}_worst": np.max(acc[m]) for m in ("full", "trunc", "ridge")}))
        print(pd.DataFrame(rows[-2:]).to_string(index=False, float_format=lambda v: f"{v:.3f}"),
              flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.outdir, exist_ok=True)
    out = f"{a.outdir}/truncation_test_{a.tag}.csv"
    df.to_csv(out, index=False)
    print(f"\n{df.to_string(index=False, float_format=lambda v: f'{v:.3f}')}\nSaved {out}")


if __name__ == "__main__":
    main()
