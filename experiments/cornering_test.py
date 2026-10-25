"""Cornering test on the existing 50.28 + 50.34 ensembles (left side only, y <= 0).

NOTE: these ensembles were aligned onto the reference mesh by NEAREST-NEIGHBOUR,
not IDW, and the ride-height (straight) and cornering (RRF) cases come from
different simulation set-ups. Treat the numbers as a first test, not final.

Reports, at 10 and 24 sensors, for greedy, greedy with min_dist, QR (pivoted)
and random placement, plus the "no sensors" mean-field baseline:

  (a) modes for the variance target and leave-one-out (LOO) RMS on all cases
  (b) train on the ride-height cases, test on each cornering case
  (c) LOO within the cornering cases only

Usage:
    python experiments/cornering_test.py data/side_diffuser_ensemble.npz \
        --params data/case_params_combined.csv
    python experiments/cornering_test.py data/side_diffuser_ensemble_highres_NN.npz \
        --params data/case_params_highres.csv

The script prints every array in the npz and the params columns. It refuses to
guess when the number of params rows differs from the number of ensemble
columns, or when it cannot tell cornering cases from straight ones; in that
case pass --corner-mask or --n-ride explicitly (--n-ride assumes the first
n columns are ride-height cases and the rest cornering).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr

SENSORS = (10, 24)
METHODS = ("greedy", "greedy_md", "qr", "random")
N_RANDOM = 20


# ---------------------------------------------------------------- loading
def describe_npz(z):
    print("Arrays in npz:")
    for k in z.files:
        print(f"  {k:30s} shape {z[k].shape} dtype {z[k].dtype}")


def find_matrix(z, key):
    if key:
        return z[key]
    c = [k for k in z.files if z[k].ndim == 2 and min(z[k].shape) > 3 and z[k].dtype.kind == "f"
         and max(z[k].shape) > 1000]
    if len(c) != 1:
        raise SystemExit(f"Cannot pick the Cp matrix from {c}; pass --key")
    print(f"Using '{c[0]}' as the Cp matrix")
    return z[c[0]]


def find_coords(z, N, key):
    if key:
        return z[key]
    if all(k in z.files for k in ("x", "y", "z")):
        return np.column_stack([z["x"], z["y"], z["z"]])
    for k in z.files:
        a = z[k]
        if a.ndim == 2 and a.shape[0] == N and a.shape[1] == 3:
            print(f"Using '{k}' as coordinates")
            return a
    raise SystemExit("No coordinates found (need x,y,z arrays or an N x 3 array); pass --coords-key")


def corner_mask(params, M, a):
    if len(params) != M:
        raise SystemExit(
            f"AMBIGUOUS: params has {len(params)} rows but the ensemble has {M} cases. "
            "Match by name/row order yourself, or subset the params file, then rerun.")
    if a.n_ride is not None:
        m = np.zeros(M, bool)
        m[a.n_ride:] = True
        print(f"Assuming first {a.n_ride} columns = ride-height, last {M - a.n_ride} = cornering (--n-ride)")
        return m
    if a.corner_mask:
        return pd.read_csv(a.corner_mask, header=None).iloc[:, 0].astype(bool).to_numpy()
    low = {c.lower(): c for c in params.columns}
    for name in ("cr", "r", "corner_radius", "radius"):
        if name in low:
            v = params[low[name]].to_numpy(float)
            m = np.isfinite(v) & (v > 0) & (v < 400)  # 500 m = straight line
            if 0 < m.sum() < M:
                print(f"Cornering = column '{low[name]}' < 400 m: {m.sum()} cornering, {(~m).sum()} ride-height")
                return m
    raise SystemExit("AMBIGUOUS: cannot tell cornering from straight cases; use --n-ride or --corner-mask")


# ---------------------------------------------------------------- method
def pod(Xtr, var):
    """POD via the small M x M Gram matrix (same modes as thin SVD, far faster)."""
    mean = Xtr.mean(axis=1, keepdims=True)
    Xc = Xtr - mean
    w, V = np.linalg.eigh(Xc.T @ Xc)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    r = int(np.searchsorted(np.cumsum(w) / np.sum(w), var) + 1)
    r = min(r, int((w > 1e-12 * w[0]).sum()))
    U = (Xc @ V[:, :r]) / np.sqrt(w[:r])
    return mean, U, r


def greedy(U, k, coords=None, min_dist=0.0, eps=1e-10):
    n, r = U.shape
    sel, avail, D = [], np.ones(n, bool), np.eye(r) / eps
    for _ in range(k):
        g = 1.0 + ((U @ D) * U).sum(axis=1)
        g[~avail] = -np.inf
        b = int(np.argmax(g))
        if not np.isfinite(g[b]):
            break
        sel.append(b)
        avail[b] = False
        if min_dist > 0:
            avail &= np.linalg.norm(coords - coords[b], axis=1) >= min_dist
        w = U[b][:, None]
        Dw = D @ w
        D -= Dw @ Dw.T / (1.0 + (w.T @ Dw).item())
    return sel


def place(method, U, k, coords, md, rng):
    if method == "greedy":
        return greedy(U, k)
    if method == "greedy_md":
        return greedy(U, k, coords, md)
    if method == "qr":
        return list(qr(U.T, pivoting=True, mode="economic")[2][:k])
    return list(rng.choice(U.shape[0], k, replace=False))


def evaluate(X, tr, te, k, var, coords, md, seed=0):
    """Return {method: [rms per test case]}, baseline list, modes."""
    mean, U, r = pod(X[:, tr], var)
    rng = np.random.default_rng(seed)
    out = {}
    for m in METHODS:
        if m == "random":
            sels = [place(m, U, k, coords, md, rng) for _ in range(N_RANDOM)]
        else:
            sels = [place(m, U, k, coords, md, rng)]
        errs = []
        for j in te:
            e = []
            for s in sels:
                a, *_ = np.linalg.lstsq(U[s], X[s, j] - mean[s, 0], rcond=None)
                e.append(np.sqrt(np.mean((mean[:, 0] + U @ a - X[:, j]) ** 2)))
            errs.append(np.mean(e))
        out[m] = errs
    base = [np.sqrt(np.mean((mean[:, 0] - X[:, j]) ** 2)) for j in te]
    return out, base, r


def run_loo(X, idx, k, var, coords, md):
    res = {m: [] for m in METHODS}
    base, modes = [], []
    for j in idx:
        tr = [i for i in idx if i != j]
        o, b, r = evaluate(X, tr, [j], k, var, coords, md)
        for m in METHODS:
            res[m] += o[m]
        base += b
        modes.append(r)
    return res, base, modes


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--params", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--coords-key", default=None)
    p.add_argument("--n-ride", type=int, default=None)
    p.add_argument("--corner-mask", default=None, help="CSV, one 0/1 per case")
    p.add_argument("--variance", type=float, default=0.90)
    p.add_argument("--min-dist", type=float, default=0.05, help="metres")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()

    z = np.load(a.ensemble, allow_pickle=True)
    describe_npz(z)
    X = find_matrix(z, a.key).astype(float)
    N, M = X.shape if X.shape[0] > X.shape[1] else X.T.shape
    if X.shape[0] < X.shape[1]:
        X = X.T
    coords = find_coords(z, X.shape[0], a.coords_key).astype(float)
    params = pd.read_csv(a.params)
    print(f"\nParams columns: {list(params.columns)} ({len(params)} rows)")
    print(params.head(3).to_string())
    print(params.tail(3).to_string())

    ext = np.ptp(coords, axis=0).max()
    md = a.min_dist * (1000.0 if ext > 20 else 1.0)  # coords in mm if extent > 20
    print(f"Coordinate extent {ext:.1f} -> treating as {'mm' if ext > 20 else 'm'}; min_dist = {md:g}")

    keep = coords[:, 1] <= 0
    print(f"Keeping y <= 0: {keep.sum()} of {len(keep)} points")
    X, coords = X[keep], coords[keep]
    M = X.shape[1]
    cm = corner_mask(params, M, a)
    ride, corner = np.where(~cm)[0], np.where(cm)[0]
    print(f"{len(ride)} ride-height cases, {len(corner)} cornering cases, {M} total")
    print("NOTE: aligned by nearest-neighbour (not IDW); straight and cornering cases are different set-ups.\n")

    rows = []

    def add(test, k, res, base, modes):
        for m in METHODS:
            rows.append(dict(test=test, sensors=k, method=m, mean_rms=np.mean(res[m]),
                             worst_rms=np.max(res[m]), n_cases=len(res[m]), modes_median=int(np.median(modes))))
        rows.append(dict(test=test, sensors=k, method="mean_field_baseline", mean_rms=np.mean(base),
                         worst_rms=np.max(base), n_cases=len(base), modes_median=int(np.median(modes))))

    for k in SENSORS:
        res, base, modes = run_loo(X, list(range(M)), k, a.variance, coords, md)
        add("a_LOO_all", k, res, base, modes)
        res, base, r = evaluate(X, list(ride), list(corner), k, a.variance, coords, md)
        add("b_train_ride_test_corner", k, res, base, [r])
        res, base, modes = run_loo(X, list(corner), k, a.variance, coords, md)
        add("c_LOO_corner_only", k, res, base, modes)

    df = pd.DataFrame(rows)
    os.makedirs(a.outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.ensemble))[0]
    df.to_csv(f"{a.outdir}/cornering_test_{tag}.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    names = METHODS + ("mean_field_baseline",)
    for ax, t in zip(axs, ("a_LOO_all", "b_train_ride_test_corner", "c_LOO_corner_only")):
        w = 0.38
        for i, k in enumerate(SENSORS):
            d = df[(df.test == t) & (df.sensors == k)].set_index("method").loc[list(names)]
            x = np.arange(len(names)) + (i - 0.5) * w
            ax.bar(x, d.mean_rms, w, label=f"{k} sensors")
            ax.plot(x, d.worst_rms, "k_", ms=6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["greedy", "greedy\nmin_dist", "QR", "random", "no\nsensors"], fontsize=7)
        ax.set_title(t.replace("_", " "), fontsize=8)
    axs[0].set_ylabel("RMS Cp error (bar mean, tick worst)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Nearest-neighbour-aligned ensemble, y <= 0", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{a.outdir}/cornering_test_{tag}.png", dpi=200)
    print(f"\nSaved {a.outdir}/cornering_test_{tag}.csv and .png")


if __name__ == "__main__":
    main()
