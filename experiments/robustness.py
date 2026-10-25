"""Robustness checks for POD + greedy sensing + least-squares reconstruction.

Data and conventions follow experiments/ml_reconstruction.py:
side_diffuser_ensemble_highres_NN.npz (aligned by nearest neighbour) with
case_params.csv, y <= 0 only, POD at 90 % variance (pod() from
cornering_test.py), spaced greedy sensors with min_dist 0.05 m (greedy() from
cornering_test.py), lstsq reconstruction, RMS Cp error over all points.

Split B: train on the straight cases, test on each cornering case. POD and
sensors are always refitted on the training cases only.

Parts (--part):
  2  failure modes: (2a) single-sensor dropout, (2b) unsupported-input flag
  3  uncertainty propagation of iid reading noise (MMA3001 Week 2)
  4  numerical precision: Gram vs SVD POD, float32 vs float64 (MMA3001 Weeks 2 and 9)
  8  input resolution: greedy on candidate subsets, POD cost against point count

Each part writes results/robustness_<part>.csv and results/robustness_<part>.png.

Usage:
    python experiments/robustness.py --part all
    python experiments/robustness.py --part 3
"""
import argparse
import contextlib
import io
import os
import sys
import time
import tracemalloc
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cornering_test import corner_mask, greedy, pod  # noqa: E402

VARIANCE = 0.90
MIN_DIST = 0.05  # m
SIGMA = 0.03     # iid reading noise in Cp (part 3)
OUT = "results"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


# ---------------------------------------------------------------- shared helpers
def load(ensemble="data/side_diffuser_ensemble_highres_NN.npz", params_path="data/case_params.csv"):
    z = np.load(ensemble)
    X = z["Cp_matrix"].astype(float)
    coords = np.column_stack([z["x"], z["y"], z["z"]]).astype(float)
    params = pd.read_csv(params_path)
    assert len(params) == X.shape[1], f"{len(params)} params rows but {X.shape[1]} ensemble columns"
    keep = coords[:, 1] <= 0
    X, coords = X[keep], coords[keep]
    with contextlib.redirect_stdout(io.StringIO()):  # corner_mask prints its decision
        cm = corner_mask(params, X.shape[1], types.SimpleNamespace(n_ride=None, corner_mask=None))
    return types.SimpleNamespace(X=X, coords=coords, straight=np.where(~cm)[0], corner=np.where(cm)[0],
                                 names=params["folder"].to_numpy())


def rebuild(U, mean, s, x):
    """lstsq reconstruction from readings x[s]; returns the field and the coefficients."""
    a = np.linalg.lstsq(U[s], x[s] - mean[s, 0], rcond=None)[0]
    return mean[:, 0] + U @ a, a


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def residual(U, mean, s, x):
    """Sensor-space misfit ||y_s - mean_s - U_s a|| / sqrt(k) after the lstsq fit."""
    _, a = rebuild(U, mean, s, x)
    return float(np.linalg.norm(x[s] - mean[s, 0] - U[s] @ a) / np.sqrt(len(s)))


def predicted_noise_rms(U, s, sigma):
    """Noise-only field RMS for iid reading noise of s.d. sigma (U with orthonormal columns).

    sigma * sqrt(trace(inv(U_s^T U_s)) / n). When k < r, U_s^T U_s is singular and lstsq
    returns the minimum-norm solution, whose noise is sigma * ||pinv(U_s)||_F / sqrt(n);
    the two expressions agree whenever k >= r.
    """
    Us = U[s]
    if len(s) >= U.shape[1]:
        tr = np.trace(np.linalg.inv(Us.T @ Us))
    else:
        sv = np.linalg.svd(Us, compute_uv=False)
        tr = np.sum(1.0 / sv ** 2)
    return float(sigma * np.sqrt(tr / U.shape[0]))


def monte_carlo_noise_rms(U, mean, s, fields, sigma, draws, rng):
    """sqrt(mean(rms_noisy^2 - rms_exact^2)) over draws and the columns of fields."""
    Us, extra = U[s], []
    for x in fields.T:
        exact, _ = rebuild(U, mean, s, x)
        Y = x[s][:, None] + sigma * rng.standard_normal((len(s), draws)) - mean[s]
        A = np.linalg.lstsq(Us, Y, rcond=None)[0]
        err = mean + U @ A - x[:, None]
        extra.append(np.mean(err ** 2, axis=0) - rms(exact, x) ** 2)
    return float(np.sqrt(np.mean(extra)))


def min_spacing(coords, s):
    p = coords[s]
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    return float(d[np.triu_indices(len(s), 1)].min())


def pod_svd(Xc):
    """Left singular vectors and singular values of a centred matrix by thin SVD."""
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    return U, s


def pod_gram(Xc):
    """Same via the eigendecomposition of the small Gram matrix Xc^T Xc, U = Xc V / s."""
    w, V = np.linalg.eigh(Xc.T @ Xc)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    s = np.sqrt(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        U = (Xc @ V) / s
    return U, s


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, part):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"robustness_{part}.png"), dpi=300, facecolor=SURFACE)
    plt.close(fig)


def show(title, df):
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


# ---------------------------------------------------------------- part 2
def part2(D):
    X, coords, M = D.X, D.coords, D.X.shape[1]
    rows = []

    # 2a: single-sensor dropout, split B and LOO over all cases
    folds = [("B", D.straight, D.corner)] + [("LOO", np.delete(np.arange(M), j), np.array([j])) for j in range(M)]
    for split, tr, te in folds:
        mean, U, r = pod(X[:, tr], VARIANCE)
        s_all = greedy(U, 24, coords, MIN_DIST)  # greedy is sequential: first 10 = a 10-sensor run
        for k in (10, 24):
            s = s_all[:k]
            for j in te:
                x = X[:, j]
                drops = [rms(rebuild(U, mean, s[:i] + s[i + 1:], x)[0], x) for i in range(k)]
                i = int(np.argmax(drops))
                rows.append(dict(section="2a", split="B" if split == "B" else "LOO", k=k, case=j,
                                 case_name=D.names[j], modes=r, rms_all=rms(rebuild(U, mean, s, x)[0], x),
                                 rms_drop_mean=np.mean(drops), worst_sensor_rank=i + 1, worst_point=s[i],
                                 worst_x_mm=coords[s[i], 0] * 1e3, worst_y_mm=coords[s[i], 1] * 1e3,
                                 worst_z_mm=coords[s[i], 2] * 1e3, rms_worst_drop=drops[i]))

    # 2b: residual flag, threshold from LOO within the straight cases
    ks = (24, 32, 40)
    for j in D.straight:
        tr = D.straight[D.straight != j]
        mean, U, r = pod(X[:, tr], VARIANCE)
        s_all = greedy(U, max(ks), coords, MIN_DIST)
        for k in ks:
            s = s_all[:k]
            rows.append(dict(section="2b", split="straight LOO", k=k, case=j, case_name=D.names[j], modes=r,
                             residual=residual(U, mean, s, X[:, j]), rms_all=rms(rebuild(U, mean, s, X[:, j])[0], X[:, j])))
    mean, U, r = pod(X[:, D.straight], VARIANCE)
    s_all = greedy(U, max(ks), coords, MIN_DIST)
    for k in ks:
        s = s_all[:k]
        for j in D.corner:
            rows.append(dict(section="2b", split="B cornering", k=k, case=j, case_name=D.names[j], modes=r,
                             residual=residual(U, mean, s, X[:, j]), rms_all=rms(rebuild(U, mean, s, X[:, j])[0], X[:, j])))

    df = pd.DataFrame(rows)
    b = df[df.section == "2b"].copy()
    thr = b[b.split == "straight LOO"].groupby("k")["residual"].quantile(0.95)
    b["threshold"] = b["k"].map(thr)
    b["flagged"] = b["residual"] > b["threshold"]
    df.loc[b.index, ["threshold", "flagged"]] = b[["threshold", "flagged"]]
    df.to_csv(os.path.join(OUT, "robustness_2.csv"), index=False)

    a = df[df.section == "2a"]
    summ_a = []
    for (split, k), g in a.groupby(["split", "k"], sort=False):
        w = g.loc[g.rms_worst_drop.idxmax()]
        summ_a.append(dict(split=split, k=k, rms_all=g.rms_all.mean(), rms_drop_mean=g.rms_drop_mean.mean(),
                           worst_point=int(w.worst_point), x_mm=w.worst_x_mm, y_mm=w.worst_y_mm, z_mm=w.worst_z_mm,
                           worst_rms=w.rms_worst_drop, worst_case=w.case_name.split("_")[0]))
    show("2a  single-sensor dropout (means over test cases; worst = largest single-dropout RMS)", pd.DataFrame(summ_a))

    summ_b = []
    for k, g in b.groupby("k"):
        st, co = g[g.split == "straight LOO"], g[g.split == "B cornering"]
        rho = spearmanr(g.residual, g.rms_all).statistic
        summ_b.append(dict(k=k, threshold=thr[k], corner_flagged=co.flagged.mean(),
                           straight_flagged=st.flagged.mean(), spearman_resid_vs_rms=rho, modes=int(g.modes.median())))
    show("2b  unsupported-input flag (threshold = 95th pct of straight LOO residuals)", pd.DataFrame(summ_b))

    fig, ax = plt.subplots(figsize=(3.45, 2.8), facecolor=SURFACE)
    g = b[b.k == 24]
    st, co = g[g.split == "straight LOO"], g[g.split == "B cornering"]
    ax.scatter(st.rms_all, st.residual, s=12, color=BLUE, marker="o", label="straight (LOO)", linewidths=0)
    ax.scatter(co.rms_all, co.residual, s=16, color=ORANGE, marker="^", label="cornering (split B)", linewidths=0)
    ax.axhline(thr[24], color=INK2, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], thr[24], " 95th pct", va="bottom", ha="right", fontsize=6.5, color=INK2)
    ax.set_xlabel("True RMS error in Cp", fontsize=7, color=INK2)
    ax.set_ylabel("Sensor residual", fontsize=7, color=INK2)
    ax.set_title("Unsupported-input flag, 24 sensors", fontsize=7.5, loc="left", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    style(ax)
    save(fig, 2)


# ---------------------------------------------------------------- part 3
def part3(D, draws=200):
    X, coords = D.X, D.coords
    ks = (4, 6, 8, 10, 13, 16, 20, 24, 32, 40, 60, 80)
    mean, U, r = pod(X[:, D.straight], VARIANCE)
    assert np.linalg.norm(U.T @ U - np.eye(r)) < 1e-8, "U must have orthonormal columns"
    n = U.shape[0]
    layouts = {("greedy", -1): greedy(U, max(ks), coords, MIN_DIST)}
    for seed in range(5):
        layouts[("random", seed)] = list(np.random.default_rng(seed).choice(n, max(ks), replace=False))
    rng = np.random.default_rng(12345)
    rows = []
    for (placement, seed), sel in layouts.items():
        for k in ks:
            if k > len(sel):
                rows.append(dict(placement=placement, seed=seed, k=k, modes=r, note=f"only {len(sel)} spaced sensors fit"))
                continue
            s = sel[:k]
            Us = U[s]
            rows.append(dict(placement=placement, seed=seed, k=k, modes=r,
                             predicted_noise_rms=predicted_noise_rms(U, s, SIGMA),
                             mc_noise_rms=monte_carlo_noise_rms(U, mean, s, X[:, D.corner], SIGMA, draws, rng),
                             cond_Us=np.linalg.cond(Us), logdet_UsTUs=np.linalg.slogdet(Us.T @ Us)[1],
                             mean_rms_noise_free=np.mean([rms(rebuild(U, mean, s, X[:, j])[0], X[:, j]) for j in D.corner])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "robustness_3.csv"), index=False)

    summ = summarise_part3(df)
    summ["ratio_mc_pred"] = summ.monte_carlo / summ.predicted
    show(f"3  noise propagation, sigma = {SIGMA}, r = {r} modes, {draws} draws x {len(D.corner)} cornering cases "
         "(random = mean of 5 seeds)", summ)
    if "note" in df and df.note.notna().any():
        print("   " + "; ".join(sorted(set(f"{p} k={k}: {t}" for p, k, t in df[df.note.notna()][["placement", "k", "note"]].values))))
    plot_part3(summ, r, ks)


def summarise_part3(df):
    ok = df.dropna(subset=["predicted_noise_rms"])
    return (ok.groupby(["placement", "k"]).agg(predicted=("predicted_noise_rms", "mean"), monte_carlo=("mc_noise_rms", "mean"),
                                               cond_Us=("cond_Us", "median"), logdet=("logdet_UsTUs", "median"))
            .reset_index())


def plot_part3(summ, r, ks):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True, facecolor=SURFACE)
    for placement, colour in (("greedy", BLUE), ("random", ORANGE)):
        g = summ[summ.placement == placement]
        ax1.plot(g.k, g.predicted, color=colour, lw=1.5, label=f"{placement}, predicted")
        ax1.plot(g.k, g.monte_carlo, color=colour, lw=0, marker="o", ms=3.5, label=f"{placement}, Monte Carlo")
        ax2.plot(g.k, g.cond_Us, color=colour, lw=1.5, marker="o", ms=3, label=placement)
    for ax in (ax1, ax2):
        ax.axvline(r, color=INK2, lw=1, ls="--")
        style(ax)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax2.text(r, ax2.get_ylim()[0], f" k = r = {r}", fontsize=6.5, color=INK2, va="bottom")
    ax1.set_ylabel("Noise RMS in Cp", fontsize=7, color=INK2)
    ax2.set_ylabel("cond(U_s)", fontsize=7, color=INK2)
    ax2.set_xlabel("Sensors k", fontsize=7, color=INK2)
    ax2.set_xticks(ks, [str(k) for k in ks], fontsize=6)
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.legend(fontsize=6, frameon=False, loc="upper right")
    ax1.set_title(f"Reading noise sigma = {SIGMA}, split B", fontsize=7.5, loc="left", color=INK)
    save(fig, 3)


# ---------------------------------------------------------------- part 4
def timed_mem(fn, *args):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args)
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, dt, peak


def split_b_rms(U, mean, coords, X, corner, k=24):
    s = greedy(U, k, coords, MIN_DIST)
    return s, float(np.mean([rms(rebuild(U, mean, s, X[:, j])[0], X[:, j]) for j in corner]))


def part4(D):
    X, coords = D.X, D.coords
    Xtr = X[:, D.straight]
    mean = Xtr.mean(axis=1, keepdims=True)
    Xc = Xtr - mean
    (U_s, s_s), t_svd, m_svd = timed_mem(pod_svd, Xc)
    (U_g, s_g), t_gram, m_gram = timed_mem(pod_gram, Xc)
    r = int(np.searchsorted(np.cumsum(s_s ** 2) / np.sum(s_s ** 2), VARIANCE) + 1)
    angle = float(np.max(subspace_angles(U_g[:, :r], U_s[:, :r])))
    sel_s, rms_s = split_b_rms(U_s[:, :r], mean, coords, X, D.corner)
    sel_g, rms_g = split_b_rms(U_g[:, :r], mean, coords, X, D.corner)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(s_g - s_s) / s_s
    expected = np.finfo(float).eps * (s_s[0] / s_s) ** 2
    rows = [dict(section="4a", index=i + 1, s_svd=s_s[i], s_gram=s_g[i], rel_error=rel[i], expected_eps_kappa2=expected[i])
            for i in range(len(s_s))]
    a_summary = dict(modes=r, max_principal_angle_rad=angle, max_principal_angle_deg=np.degrees(angle),
                     rms_k24_svd=rms_s, rms_k24_gram=rms_g, rms_diff=rms_g - rms_s,
                     sensor_overlap=len(set(sel_s) & set(sel_g)),
                     time_svd_s=t_svd, time_gram_s=t_gram, peak_mem_svd_MB=m_svd / 2**20, peak_mem_gram_MB=m_gram / 2**20)
    rows.append(dict(section="4a summary", **a_summary))

    b = {}
    for dtype in (np.float64, np.float32):
        Xd = X.astype(dtype)
        t0 = time.perf_counter()
        mean_d, U_d, r_d = pod(Xd[:, D.straight], VARIANCE)
        s_d = greedy(U_d, 24, coords.astype(dtype), MIN_DIST)
        recon = [rebuild(U_d, mean_d, s_d, Xd[:, j])[0] for j in D.corner]
        dt = time.perf_counter() - t0
        b[dtype.__name__] = dict(sel=s_d, modes=r_d, nbytes=Xd.nbytes, time=dt,
                                 rms=float(np.mean([rms(xh.astype(float), X[:, j]) for xh, j in zip(recon, D.corner)])))
    overlap = len(set(b["float64"]["sel"]) & set(b["float32"]["sel"]))
    for name, v in b.items():
        rows.append(dict(section="4b", dtype=name, modes=v["modes"], mean_rms_k24=v["rms"], X_nbytes=v["nbytes"],
                         time_s=v["time"], sensor_overlap_with_other=overlap))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "robustness_4.csv"), index=False)

    show("4a  Gram vs SVD on the split-B training matrix "
         f"({Xc.shape[0]} x {Xc.shape[1]})", pd.DataFrame([a_summary]).T.reset_index().rename(columns={"index": "quantity", 0: "value"}))
    valid = s_s > 1e-10 * s_s[0]
    print(f"   relative singular-value error: max {np.nanmax(rel[valid]):.2e} over the {valid.sum()} non-null indices "
          f"(index {len(s_s)} is the null direction of the centred matrix)")
    show("4b  float32 vs float64, split B, k = 24 "
         f"(sensor overlap {overlap} of 24)", df[df.section == "4b"][["dtype", "modes", "mean_rms_k24", "X_nbytes", "time_s"]])

    fig, ax = plt.subplots(figsize=(3.45, 2.8), facecolor=SURFACE)
    idx = np.arange(1, len(s_s) + 1)
    ax.semilogy(idx[valid], np.maximum(rel[valid], 1e-17), color=BLUE, marker="o", ms=2.5, lw=1, label="|s_gram - s_svd| / s_svd")
    ax.semilogy(idx[valid], expected[valid], color=INK2, lw=1, ls="--", label="eps (s_1 / s_i)^2")
    ax.axvline(r, color=INK2, lw=0.8, ls=":")
    ax.text(r, ax.get_ylim()[1], f" r = {r}", fontsize=6.5, color=INK2, va="top")
    ax.set_xlabel("Singular value index i", fontsize=7, color=INK2)
    ax.set_ylabel("Relative error", fontsize=7, color=INK2)
    ax.set_title("Gram vs SVD singular values, split-B training", fontsize=7.5, loc="left", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    style(ax)
    save(fig, 4)


# ---------------------------------------------------------------- part 8
def part8(D):
    X, coords = D.X, D.coords
    mean, U, r = pod(X[:, D.straight], VARIANCE)
    n, m = X.shape[0], len(D.straight)
    sizes = [5000, 10000, 20000, 40000, 80000, n]
    rows = []
    for size in sizes:
        for seed in range(3):
            cand = np.arange(n) if size == n else np.sort(np.random.default_rng(seed).choice(n, size, replace=False))
            t0 = time.perf_counter()
            sl = greedy(U[cand], 24, coords[cand], MIN_DIST)
            dt = time.perf_counter() - t0
            s = list(cand[sl])
            rows.append(dict(section="8 greedy", candidates=size, seed=seed, modes=r, greedy_time_s=dt,
                             mean_rms_k24=np.mean([rms(rebuild(U, mean, s, X[:, j])[0], X[:, j]) for j in D.corner]),
                             min_spacing_mm=min_spacing(coords, s) * 1e3))
    Xtr = X[:, D.straight]
    for size in sizes:
        Xs = Xtr if size == n else Xtr[np.sort(np.random.default_rng(0).choice(n, size, replace=False))]
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            pod(Xs, VARIANCE)
            times.append(time.perf_counter() - t0)
        rows.append(dict(section="8 pod timing", candidates=size, pod_time_s=min(times), gram_flops=2 * size * m ** 2))
    df = pd.DataFrame(rows)
    pt = df[df.section == "8 pod timing"]
    slope = float(np.polyfit(np.log(pt.candidates), np.log(pt.pod_time_s), 1)[0])
    df["loglog_slope_pod_time"] = np.where(df.section == "8 pod timing", slope, np.nan)
    df.to_csv(os.path.join(OUT, "robustness_8.csv"), index=False)

    g = (df[df.section == "8 greedy"].groupby("candidates")
         .agg(mean_rms_k24=("mean_rms_k24", "mean"), rms_sd=("mean_rms_k24", "std"), greedy_time_s=("greedy_time_s", "mean"),
              min_spacing_mm=("min_spacing_mm", "mean")).reset_index())
    g = g.merge(pt[["candidates", "pod_time_s", "gram_flops"]], on="candidates")
    show(f"8  input resolution, split B, k = 24, r = {r} (3 seeds; POD time = best of 3 on {m} training cases)", g)
    print(f"   log-log slope of POD (Gram) time against point count: {slope:.2f} (expected about 1)")

    fig, ax = plt.subplots(figsize=(3.45, 2.8), facecolor=SURFACE)
    ax2 = ax.twinx()
    ax.plot(g.candidates, g.mean_rms_k24, color=BLUE, marker="o", ms=3.5, lw=1.5, label="mean RMS (left)")
    ax2.plot(g.candidates, g.greedy_time_s, color=ORANGE, marker="s", ms=3.5, lw=1.5, label="greedy time (right)")
    ax.set_xscale("log")
    ax.set_xlabel("Greedy candidate points", fontsize=7, color=INK2)
    ax.set_ylabel("Mean RMS error in Cp, k = 24", fontsize=7, color=BLUE)
    ax2.set_ylabel("Greedy wall time [s]", fontsize=7, color=ORANGE)
    ax2.tick_params(labelsize=7, colors=INK2)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    style(ax)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=6.5, frameon=False, loc="upper center")
    ax.set_title("Input resolution, split B", fontsize=7.5, loc="left", color=INK)
    save(fig, 8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["2", "3", "4", "8", "all"], default="all")
    a = p.parse_args()
    t_all = time.perf_counter()
    D = load()
    os.makedirs(OUT, exist_ok=True)
    print(f"{D.X.shape[1]} cases ({len(D.straight)} straight, {len(D.corner)} cornering), {D.X.shape[0]} points "
          "with y <= 0; ensemble aligned by nearest neighbour")
    parts = ["3", "2", "4", "8"] if a.part == "all" else [a.part]
    for part in parts:
        t0 = time.perf_counter()
        {"2": part2, "3": part3, "4": part4, "8": part8}[part](D)
        print(f"   part {part} run time: {time.perf_counter() - t0:.1f} s")
    print(f"\nTotal run time: {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
