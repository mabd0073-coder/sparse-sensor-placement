"""Two report figures:

1. fig_mapping_nn_idw.png - how nearest-neighbour (k = 1) and inverse-distance
   weighting (k = 4, weights 1/d as in cluster/master_load_sweep_ensemble.py)
   map one case's mesh onto the reference points. Synthetic illustration:
   (a) which source points each method uses, (b) a sharp suction peak,
   (c) a thin plate where the 4 nearest points can include the other face.
2. fig_layout_views.png - the feasible layout seen from above, the side, the
   rear and in 3-D, on the mean Cp of the ensemble (appendix figure).

Usage:
    python experiments/report_extra_figures.py --ensemble data/side_diffuser_ensemble_highres_NN.npz \
        --layout results/feasible_layout.csv --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NN_C, IDW_C, TRUE_C = "#e8642c", "#2a6fdb", "0.15"


def idw(src, val, tgt, k=4, eps=1e-9):
    d, i = cKDTree(src).query(tgt, k=k)
    w = 1.0 / (d + eps)
    return (w * val[i]).sum(1) / w.sum(1), d, i, w


def mapping_figure(out):
    rng = np.random.default_rng(4)
    fig, axs = plt.subplots(3, 1, figsize=(3.45, 6.0), gridspec_kw=dict(height_ratios=[1.15, 1, 0.9]))

    # (a) which points each method uses
    ax = axs[0]
    g = np.stack(np.meshgrid(np.arange(6), np.arange(4)), -1).reshape(-1, 2).astype(float)
    src = g * 4.0 + rng.uniform(-1.3, 1.3, g.shape)
    tgt = np.array([[9.3, 6.1]])
    _, d, i, w = idw(src, np.zeros(len(src)), tgt)
    ax.scatter(*src.T, s=14, c="0.6", label="source mesh points (one case)")
    for j, (dj, ij, wj) in enumerate(zip(d[0], i[0], w[0] / w[0].sum())):
        ax.plot(*np.c_[tgt[0], src[ij]], color=IDW_C, lw=0.9, ls="--", zorder=1)
        dvec = (src[ij] - tgt[0]) / np.linalg.norm(src[ij] - tgt[0])
        ax.text(*(src[ij] + 1.1 * dvec), f"w={wj:.2f}", fontsize=6.0, color=IDW_C, ha="center", va="center")
    ax.plot(*np.c_[tgt[0], src[i[0, 0]]], color=NN_C, lw=2.2, zorder=2)
    ax.scatter(*tgt.T, marker="*", s=110, c="k", zorder=3, label="reference point")
    ax.plot([], [], color=NN_C, lw=2.2, label="nearest neighbour: copy 1 value")
    ax.plot([], [], color=IDW_C, lw=0.9, ls="--", label="IDW: 4 nearest, weights ∝ 1/d")
    ax.set_title("(a) points used for one reference point", fontsize=7.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)

    # (b) a sharp suction peak along a line
    ax = axs[1]
    f = lambda s: -0.3 - 1.4 * np.exp(-((s - 21.0) / 2.2) ** 2)
    xs = np.sort(rng.uniform(0, 40, 22))
    xt = np.linspace(0, 40, 600)
    nn = f(xs)[cKDTree(xs[:, None]).query(xt[:, None])[1]]
    iv = idw(xs[:, None], f(xs), xt[:, None])[0]
    ax.plot(xt, f(xt), color=TRUE_C, lw=1.0, label="true Cp")
    ax.plot(xt, nn, color=NN_C, lw=1.2, label="nearest neighbour")
    ax.plot(xt, iv, color=IDW_C, lw=1.2, ls="--", label="IDW, k = 4")
    ax.plot(xs, f(xs), "o", ms=2.5, c="0.5")
    ax.text(0.98, 0.62, f"peak Cp\ntrue {f(xt).min():.2f}\nNN {nn.min():.2f}\nIDW {iv.min():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0)
    ax.set_title("(b) sharp suction peak, coarse source mesh", fontsize=7.5)
    ax.set_xlabel("distance along surface [mm]", fontsize=7)
    ax.set_ylabel("Cp", fontsize=7)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.2)

    # (c) thin plate: the other face is within the 4 nearest points
    ax = axs[2]
    t = 3.0
    xu = np.arange(0, 31, 3.5) + 0.8
    xl = np.arange(0, 31, 3.5) + 2.3
    src = np.r_[np.c_[xu, np.full_like(xu, t / 2)], np.c_[xl, np.full_like(xl, -t / 2)]]
    val = np.r_[np.full(len(xu), -1.5), np.full(len(xl), 0.3)]
    tgt = np.array([[15.5, t / 2]])
    v_idw, d, i, w = idw(src, val, tgt)
    v_nn = val[cKDTree(src).query(tgt)[1]][0]
    ax.fill_between([-1, 33], -t / 2, t / 2, color="0.88")
    ax.scatter(*src[: len(xu)].T, s=12, c="#7a3db8", label="suction face, Cp = −1.5")
    ax.scatter(*src[len(xu):].T, s=12, c="#3a9a5b", label="pressure face, Cp = +0.3")
    for ij in i[0]:
        ax.plot(*np.c_[tgt[0], src[ij]], color=IDW_C, lw=0.9, ls="--")
    ax.scatter(*tgt.T, marker="*", s=90, c="k", zorder=3)
    ax.text(15.5, 4.2, f"NN {v_nn:+.2f};  IDW {v_idw[0]:+.2f} (mixes both faces)", ha="center",
            fontsize=6.3)
    ax.set_title(f"(c) thin plate ({t:.0f} mm): IDW can reach the other face", fontsize=7.5)
    ax.set_ylim(-5, 6.5); ax.set_xlim(-1, 33)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=5.8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)

    fig.tight_layout(h_pad=0.6)
    fig.savefig(out / "fig_mapping_nn_idw.png", dpi=300)


def layout_views(ensemble, layout, out, n=50000):
    z = np.load(ensemble)
    xyz = np.column_stack([z["x"], z["y"], z["z"]]).astype(float)
    if np.ptp(xyz, axis=0).max() > 20:
        xyz = xyz / 1000.0
    keep = xyz[:, 1] <= 0
    cp = z["Cp_matrix"][keep].mean(1)
    xyz = xyz[keep] * 1000
    sub = np.random.default_rng(0).choice(len(xyz), min(n, len(xyz)), replace=False)
    L = pd.read_csv(layout)
    S = L[["x_mm", "y_mm", "z_mm"]].to_numpy()

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1000, 520], height_ratios=[470, 360], wspace=0.18, hspace=0.32)
    views = [("(a) top view (from above)", 0, 1, "x [mm]", "y [mm]", gs[0, 0]),
             ("(b) rear view (looking forward)", 1, 2, "y [mm]", "z [mm]", gs[0, 1]),
             ("(c) side view (from outboard)", 0, 2, "x [mm]", "z [mm]", gs[1, 0])]
    order = np.argsort(xyz[sub, 2])
    for ttl, i, j, xl, yl, cell in views:
        ax = fig.add_subplot(cell)
        o = sub[order] if j != 2 else sub
        sc = ax.scatter(xyz[o, i], xyz[o, j], c=cp[o], s=0.4, cmap="jet", vmin=-2, vmax=1, rasterized=True)
        ax.scatter(S[:, i], S[:, j], s=24, facecolor="white", edgecolor="k", lw=0.9, zorder=3)
        for s_, p in zip(L.sensor, S):
            dy = 4 if s_ % 2 else -9
            ax.annotate(str(s_), (p[i], p[j]), xytext=(3, dy), textcoords="offset points", fontsize=5.5,
                        zorder=4, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
        ax.tick_params(labelsize=6.5)
    ax = fig.add_subplot(gs[1, 1], projection="3d")
    ax.scatter(*xyz[sub[::2]].T, c=cp[sub[::2]], s=0.3, cmap="jet", vmin=-2, vmax=1, rasterized=True)
    ax.scatter(*S.T, s=20, facecolor="white", edgecolor="k", lw=0.9, depthshade=False)
    ax.set_box_aspect(np.ptp(xyz[sub], axis=0))
    ax.view_init(elev=28, azim=-125)
    ax.set_title("(d) 3-D view", fontsize=8)
    ax.tick_params(labelsize=5)
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.018])
    fig.colorbar(sc, cax=cax, orientation="horizontal", label="mean Cp (all cases)")
    cax.tick_params(labelsize=6.5); cax.xaxis.label.set_size(7)
    fig.savefig(out / "fig_layout_views.png", dpi=250, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", required=True)
    p.add_argument("--layout", default="results/feasible_layout.csv")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mapping_figure(out)
    layout_views(a.ensemble, a.layout, out)
    print(f"Saved {out}/fig_mapping_nn_idw.png and {out}/fig_layout_views.png")


if __name__ == "__main__":
    main()
