"""Feasible, spread-out sensor layout and its validation.

Turns the greedy D-optimal layout into one that can be drilled:

1. Candidates: left side only (y <= 0), outside the CAD exclusion zones
   (`exclusion_mask.npz`, flags on the `all_points` grid, transferred to the
   ensemble points by nearest neighbour), and at least --edge-margin from a
   sharp edge, corner, open boundary or thin part.
2. Edges are found from the point cloud itself: for each point a local PCA of
   its --knn nearest neighbours gives the "surface variation"
   lambda_min / (lambda_1 + lambda_2 + lambda_3), which is ~0 on a smooth
   surface and large at creases, corners and where two faces of a thin part
   are close; an off-centre neighbourhood marks an open boundary.
3. Greedy D-optimal selection over the feasible candidates with a minimum
   spacing (--min-dist), then an exchange pass (Fedorov-type swaps) that
   replaces one sensor at a time with the feasible candidate that most
   increases log det(U_s^T U_s), keeping all constraints.
4. Validation: leave-one-out over all cases and train-on-straight /
   test-on-cornering, for the unconstrained, spaced, feasible and
   feasible+exchange layouts, each rebuilt in every fold.

Writes results/feasible_layout.csv (the layout from all cases, with the
checks for each sensor), results/feasible_layout_validation.csv and a figure.

Usage:
    python experiments/feasible_layout.py data/side_diffuser_ensemble_highres_NN.npz \
        --params data/case_params.csv --mask data/exclusion_mask.npz \
        --mask-grid data/side_diffuser_ensemble_dedup.npz --sensors 24
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from cornering_test import pod  # noqa: E402

EPS = 1e-10


# ------------------------------------------------------------ feasibility
def to_metres(xyz):
    return xyz / 1000.0 if np.ptp(xyz, axis=0).max() > 20 else xyz


def edge_metric(coords, knn, n_ref=20000, seed=0):
    """Surface variation and boundary offset for every point.

    Neighbourhoods are taken from a random subsample of n_ref points so that
    they span about 10-20 mm whatever the local mesh density; at the scale of a
    few mesh cells the two faces of a thin plate would look like an edge.
    """
    rng = np.random.default_rng(seed)
    ref = coords[rng.choice(len(coords), min(n_ref, len(coords)), replace=False)]
    d, idx = cKDTree(ref).query(coords, k=knn)
    nb = ref[idx]                                      # n x k x 3
    c = nb.mean(axis=1)
    q = nb - c[:, None, :]
    lam = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", q, q) / knn)   # ascending
    variation = lam[:, 0] / np.maximum(lam.sum(axis=1), 1e-30)
    offset = np.linalg.norm(c - coords, axis=1) / np.maximum(d.mean(axis=1), 1e-12)
    return variation, offset


def feasibility(coords, mask_xyz, mask_flags, margin, knn, var_thr, off_thr, tol=0.02):
    """Return dict of boolean arrays: left, excluded, edge, near_edge, feasible; and edge distance."""
    left = coords[:, 1] <= 0
    excluded = np.zeros(len(coords), bool)
    unknown = np.zeros(len(coords), bool)
    if mask_xyz is not None:
        dist, j = cKDTree(mask_xyz).query(coords)
        excluded = mask_flags[j] & (dist <= tol)
        unknown = dist > tol                          # no CAD point nearby: treat as infeasible
    variation, offset = edge_metric(coords, knn)
    edge = (variation > var_thr) | (offset > off_thr)
    d_edge = cKDTree(coords[edge]).query(coords)[0] if edge.any() else np.full(len(coords), np.inf)
    near_edge = d_edge < margin
    feasible = left & ~excluded & ~unknown & ~near_edge
    return dict(left=left, excluded=excluded, unknown=unknown, edge=edge, near_edge=near_edge,
                feasible=feasible, d_edge=d_edge)


# ------------------------------------------------------------ placement
def greedy(U, k, coords, allowed, min_dist):
    """Greedy D-optimal over allowed candidates with minimum spacing (vectorised)."""
    r = U.shape[1]
    avail = allowed.copy()
    D, sel = np.eye(r) / EPS, []
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


def exchange(U, sel, coords, allowed, min_dist, passes=5):
    """Swap sensors one at a time for the allowed point that most increases log det."""
    sel = list(sel)
    r = U.shape[1]
    for _ in range(passes):
        changed = False
        for i in range(len(sel)):
            others = sel[:i] + sel[i + 1:]
            A = EPS * np.eye(r) + U[others].T @ U[others]
            Ainv = np.linalg.inv(A)
            g = ((U @ Ainv) * U).sum(axis=1)           # log(1+g) = gain in log det
            ok = allowed.copy()
            ok[others] = False
            if min_dist > 0 and others:
                dmin = cKDTree(coords[others]).query(coords)[0]
                ok &= dmin >= min_dist
            g[~ok] = -np.inf
            b = int(np.argmax(g))
            if np.isfinite(g[b]) and g[b] > g[sel[i]] * (1 + 1e-9) + 1e-12 and b != sel[i]:
                sel[i] = b
                changed = True
        if not changed:
            break
    return sel


def layouts(U, k, coords, feas, min_dist):
    every = np.ones(len(U), bool)
    left = coords[:, 1] <= 0
    out = {"greedy (unconstrained)": greedy(U, k, coords, left, 0.0),
           "spaced greedy": greedy(U, k, coords, left, min_dist)}
    f = greedy(U, k, coords, feas, min_dist)
    out["feasible spaced greedy"] = f
    out["feasible + exchange"] = exchange(U, f, coords, feas, min_dist)
    del every
    return out


def rms_rebuild(U, mean, s, x):
    a, *_ = np.linalg.lstsq(U[s], x[s] - mean[s, 0], rcond=None)
    return float(np.sqrt(np.mean((mean[:, 0] + U @ a - x) ** 2)))


def logdet(U, s):
    return float(np.linalg.slogdet(EPS * np.eye(U.shape[1]) + U[s].T @ U[s])[1])


# ------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--params", required=True)
    p.add_argument("--mask", default=None, help="exclusion_mask.npz (key 'excluded')")
    p.add_argument("--mask-grid", default=None, help="npz with x, y, z of the mask's points")
    p.add_argument("--sensors", type=int, default=24)
    p.add_argument("--min-dist", type=float, default=0.05, help="m")
    p.add_argument("--edge-margin", type=float, default=0.015, help="m")
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--var-thr", type=float, default=0.12)
    p.add_argument("--off-thr", type=float, default=0.6)
    p.add_argument("--variance", type=float, default=0.90)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()

    z = np.load(a.ensemble)
    X = z["Cp_matrix"].astype(float)
    coords = to_metres(np.column_stack([z["x"], z["y"], z["z"]]).astype(float))
    keep = coords[:, 1] <= 0
    X, coords = X[keep], coords[keep]
    prm = pd.read_csv(a.params)
    assert len(prm) == X.shape[1]
    cr = prm["CR"].to_numpy(float)
    corner = (cr > 0) & (cr < 400)
    ride, cor = np.where(~corner)[0], np.where(corner)[0]

    mxyz = mflags = None
    if a.mask:
        mflags = np.load(a.mask)["excluded"].astype(bool)
        g = np.load(a.mask_grid)
        mxyz = to_metres(np.column_stack([g["x"], g["y"], g["z"]]).astype(float))
        assert len(mxyz) == len(mflags), "mask and grid lengths differ"
    F = feasibility(coords, mxyz, mflags, a.edge_margin, a.knn, a.var_thr, a.off_thr)
    n = len(coords)
    print(f"{n} points on the left side. Excluded by CAD zones: {F['excluded'].sum()} "
          f"({F['excluded'].mean():.0%}); no CAD point within 20 mm: {F['unknown'].sum()}; "
          f"edge/boundary/thin points: {F['edge'].sum()}; within {a.edge_margin * 1000:.0f} mm of one: "
          f"{F['near_edge'].sum()}. Feasible candidates: {F['feasible'].sum()} ({F['feasible'].mean():.0%})")
    feas = F["feasible"]

    # ---- final layout from all cases
    mean, U, r = pod(X, a.variance)
    L = layouts(U, a.sensors, coords, feas, a.min_dist)
    rows = []
    for name, s in L.items():
        d = np.linalg.norm(coords[s][:, None] - coords[s][None], axis=2)
        np.fill_diagonal(d, np.inf)
        print(f"  {name:25s} log det {logdet(U, s):8.2f}; closest pair {d.min() * 1000:6.1f} mm; "
              f"min edge distance {F['d_edge'][s].min() * 1000:5.1f} mm; "
              f"in exclusion zone {F['excluded'][s].sum()}; infeasible {(~feas[s]).sum()}")
    s = L["feasible + exchange"]
    d = np.linalg.norm(coords[s][:, None] - coords[s][None], axis=2)
    np.fill_diagonal(d, np.inf)
    for i, j in enumerate(s):
        rows.append(dict(sensor=i + 1, point_index=int(j), x_mm=coords[j, 0] * 1000, y_mm=coords[j, 1] * 1000,
                         z_mm=coords[j, 2] * 1000, edge_distance_mm=F["d_edge"][j] * 1000,
                         nearest_sensor_mm=d[i].min() * 1000, in_exclusion_zone=bool(F["excluded"][j]),
                         feasible=bool(feas[j])))
    os.makedirs(a.outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.outdir}/feasible_layout.csv", index=False, float_format="%.2f")
    plot(coords, F, L, f"{a.outdir}/feasible_layout.png")

    if a.no_validate:
        return
    # ---- validation, layouts rebuilt in every fold
    res = []
    M = X.shape[1]
    for j in range(M):
        tr = [i for i in range(M) if i != j]
        mean, U, r = pod(X[:, tr], a.variance)
        for name, s in layouts(U, a.sensors, coords, feas, a.min_dist).items():
            res.append(dict(test="a_LOO_all", layout=name, case=j, rms=rms_rebuild(U, mean, s, X[:, j])))
        res.append(dict(test="a_LOO_all", layout="no sensors", case=j,
                        rms=float(np.sqrt(np.mean((mean[:, 0] - X[:, j]) ** 2)))))
    mean, U, r = pod(X[:, ride], a.variance)
    for name, s in layouts(U, a.sensors, coords, feas, a.min_dist).items():
        for j in cor:
            res.append(dict(test="b_train_ride_test_corner", layout=name, case=j, rms=rms_rebuild(U, mean, s, X[:, j])))
    for j in cor:
        res.append(dict(test="b_train_ride_test_corner", layout="no sensors", case=j,
                        rms=float(np.sqrt(np.mean((mean[:, 0] - X[:, j]) ** 2)))))
    df = pd.DataFrame(res)
    summ = df.groupby(["test", "layout"])["rms"].agg(["mean", "max"]).reset_index()
    summ.to_csv(f"{a.outdir}/feasible_layout_validation.csv", index=False)
    print(f"\n{a.sensors} sensors, nearest-neighbour ensemble, left side")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def plot(coords, F, L, out):
    """Top view (x-y) and side view (x-z) of candidates, infeasible regions and two layouts."""
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.7))
    sub = np.random.default_rng(0).choice(len(coords), min(25000, len(coords)), replace=False)
    for ax, (i, j), lab in zip(axs, ((0, 1), (0, 2)), ("y [m]", "z [m]")):
        f = F["feasible"][sub]
        ax.scatter(coords[sub][~f, i], coords[sub][~f, j], s=0.2, c="#f6cfcd", lw=0, rasterized=True,
                   label="excluded (CAD zone, edge or unmatched)")
        ax.scatter(coords[sub][f, i], coords[sub][f, j], s=0.4, c="0.45", lw=0, rasterized=True,
                   label="feasible candidate")
        s0 = L["greedy (unconstrained)"]
        ax.scatter(coords[s0, i], coords[s0, j], s=10, marker="x", c="#555555", lw=0.8, label="greedy (unconstrained)")
        s1 = L["feasible + exchange"]
        ax.scatter(coords[s1, i], coords[s1, j], s=12, c="#2a6fdb", edgecolor="k", lw=0.3,
                   label="feasible, spaced + exchange")
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel(lab, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_aspect("equal")
    axs[0].set_title("(a) top view", fontsize=7.5)
    axs[1].set_title("(b) side view", fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.8, frameon=False, loc="lower center", ncol=2, markerscale=1.5, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=250)


if __name__ == "__main__":
    main()
