"""Machine-learning reconstruction of the side-diffuser Cp field from sparse sensors.

Each model predicts the r POD coefficients a from k sensor readings, and the
field is rebuilt as x_hat = mean + U a. The error is the RMS Cp error over all
points (y <= 0) against the case's own CFD field.

Per fold, everything is fitted on the training cases only: POD (90 % variance,
pod() from cornering_test.py), spaced greedy sensors (greedy() with min_dist
0.05 m), the input scaler, and the targets a = U^T (x - mean). Targets are
divided by their training standard deviation and multiplied back after
prediction.

Models: mean-field baseline, lstsq (the current method), linear regression,
and an MLP. Tests at k = 10 and 24: (a) leave-one-out over all cases, (b) train
on the straight cases, test on each cornering case. An MLP sensitivity study
runs on test (b) only.

The ensemble (side_diffuser_ensemble_highres_NN.npz) was aligned onto the
reference mesh by nearest neighbour, not IDW.

Usage:
    python experiments/ml_reconstruction.py
    python experiments/ml_reconstruction.py --tests b      # test (b) and sensitivity only
"""
import argparse
import contextlib
import io
import os
import sys
import time
import types
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cornering_test import corner_mask, greedy, pod  # noqa: E402

SENSORS = (10, 24)
VARIANCE = 0.90
MIN_DIST = 0.05  # m
MODELS = ("baseline", "lstsq", "linreg", "mlp")


def make_mlp(hidden=(20,), alpha=0.001, random_state=42):
    # MLP settings from MMA3001 Notebook 5.4.
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=alpha,
        learning_rate_init=0.01, max_iter=2500, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=35, tol=1e-5, random_state=random_state))


def fit_predict(model, R_tr, A_tr, R_te):
    """Fit on scaled targets, return unscaled predictions, n_iter_ and ConvergenceWarning count."""
    sd = A_tr.std(axis=0)
    sd[sd == 0] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(R_tr, A_tr / sd)
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    n_iter = getattr(model[-1], "n_iter_", np.nan)
    return model.predict(R_te).reshape(len(R_te), -1) * sd, n_iter, n_warn


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_split(X, coords, tr, te, ks, mlp_configs=None):
    """Fit POD, sensors and models on cases tr; evaluate each case in te.

    mlp_configs: {label: kwargs for make_mlp}; default is the single Notebook 5.4 model.
    Returns rows, the r used, and the fitted MLPs keyed by (label, k).
    """
    mlp_configs = mlp_configs or {"mlp": {}}
    mean, U, r = pod(X[:, tr], VARIANCE)
    # Greedy is sequential, so the first k of a max(ks) run equal a k-sensor run.
    s_all = greedy(U, max(ks), coords, MIN_DIST)
    A_tr = (U.T @ (X[:, tr] - mean)).T
    rows, fitted = [], {}
    for k in ks:
        s = s_all[:k]
        R_tr, R_te = X[s][:, tr].T, X[s][:, te].T
        preds = {"baseline": (np.zeros((len(te), r)), np.nan, 0)}
        a_ls = np.linalg.lstsq(U[s], X[s][:, te] - mean[s], rcond=None)[0].T
        preds["lstsq"] = (a_ls, np.nan, 0)
        preds["linreg"] = fit_predict(make_pipeline(StandardScaler(), LinearRegression()), R_tr, A_tr, R_te)
        for label, kw in mlp_configs.items():
            model = make_mlp(**kw)
            preds[label] = fit_predict(model, R_tr, A_tr, R_te)
            fitted[(label, k)] = model
        for model_name, (a_hat, n_iter, n_warn) in preds.items():
            for i, j in enumerate(te):
                x_hat = mean[:, 0] + U @ a_hat[i]
                rows.append(dict(k=k, case=j, model=model_name, rms=rms(x_hat, X[:, j]), modes=r,
                                 n_iter=n_iter, convergence_warnings=n_warn))
    return rows, r, fitted


def plot_bars(summary, out):
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colours = {"a_loo": "#2a78d6", "b_straight_to_corner": "#eb6834"}
    labels = {"a_loo": "(a) leave-one-out", "b_straight_to_corner": "(b) straight → cornering"}
    d = summary[summary.k == 24]
    fig, ax = plt.subplots(figsize=(3.45, 2.8), facecolor=surface)
    ax.set_facecolor(surface)
    x = np.arange(len(MODELS))
    w = 0.38
    for i, test in enumerate(colours):
        v = d[d.test == test].set_index("model").reindex(MODELS)["mean_rms"]
        ax.bar(x + (i - 0.5) * w, v, w - 0.03, color=colours[test], label=labels[test])
    ax.set_xticks(x, ["mean\nfield", "lstsq", "linear\nregr.", "MLP"], fontsize=7, color=ink)
    ax.set_ylabel("Mean RMS error in Cp", fontsize=7, color=ink2)
    ax.tick_params(axis="y", labelsize=7, colors=ink2)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_title("24 sensors, NN-aligned ensemble, y ≤ 0", fontsize=7.5, color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor=surface)
    plt.close(fig)


def plot_loss(model, out):
    ink2, surface = "#52514e", "#fcfcfb"
    mlp = model[-1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.4), sharex=True, facecolor=surface)
    epochs = np.arange(1, len(mlp.loss_curve_) + 1)
    ax1.plot(epochs, mlp.loss_curve_, color="#2a78d6", lw=1.5)
    ax1.set_ylabel("Training loss", fontsize=7, color=ink2)
    ax1.set_yscale("log")
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.plot(epochs, mlp.validation_scores_, color="#2a78d6", lw=1.5)
    ax2.set_ylabel("Validation R²", fontsize=7, color=ink2)
    ax2.set_xlabel("Epoch", fontsize=7, color=ink2)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.tick_params(labelsize=7, colors=ink2)
        ax.grid(color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.set_title("MLP (20,), test (b), 24 sensors", fontsize=7.5, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor=surface)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", default="data/side_diffuser_ensemble_highres_NN.npz")
    p.add_argument("--params", default="data/case_params.csv")
    p.add_argument("--tests", default="ba", help="which tests to run: 'a', 'b' or 'ba'")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()
    t0 = time.perf_counter()

    z = np.load(a.ensemble)
    X = z["Cp_matrix"].astype(float)
    coords = np.column_stack([z["x"], z["y"], z["z"]]).astype(float)
    params = pd.read_csv(a.params)
    assert len(params) == X.shape[1], f"{len(params)} params rows but {X.shape[1]} ensemble columns"
    keep = coords[:, 1] <= 0
    X, coords = X[keep], coords[keep]
    M = X.shape[1]
    with contextlib.redirect_stdout(io.StringIO()):  # corner_mask prints its decision
        cm = corner_mask(params, M, types.SimpleNamespace(n_ride=None, corner_mask=None))
    straight, corner = np.where(~cm)[0], np.where(cm)[0]
    names = params["folder"].to_numpy()

    rows, sens_rows = [], []
    if "b" in a.tests:
        configs = {f"mlp h={h} alpha={al:g} seed=42": dict(hidden=h, alpha=al)
                   for h in ((5,), (20,), (30, 15)) for al in (1e-3, 1e-1)}
        configs.update({f"mlp h=(20,) alpha=0.001 seed={s}": dict(random_state=s) for s in range(5)})
        configs["mlp"] = {}
        out, r, fitted = run_split(X, coords, straight, corner, SENSORS, configs)
        for row in out:
            row["test"] = "b_straight_to_corner"
            (rows if row["model"] in MODELS else sens_rows).append(row)
        plot_loss(fitted[("mlp", 24)], os.path.join(a.outdir, "ml_loss_curve.png"))
    if "a" in a.tests:
        for j in range(M):
            out, _, _ = run_split(X, coords, np.delete(np.arange(M), j), np.array([j]), SENSORS)
            for row in out:
                row["test"] = "a_loo"
                rows.append(row)

    folds = pd.DataFrame(rows)
    folds["case_name"] = names[folds["case"]]
    folds = folds[["test", "k", "case", "case_name", "model", "rms", "modes", "n_iter", "convergence_warnings"]]
    summary = (folds.groupby(["test", "k", "model"], sort=False)
               .agg(mean_rms=("rms", "mean"), worst_rms=("rms", "max"), modes=("modes", "median"),
                    n_cases=("case", "nunique"), mlp_n_iter_median=("n_iter", "median"),
                    convergence_warnings=("convergence_warnings", "sum"))
               .reset_index())

    sens = pd.DataFrame(sens_rows)
    if len(sens):
        sens["case_name"] = names[sens["case"]]
        per_model = (sens.groupby(["k", "model"]).agg(mean_rms=("rms", "mean"), worst_rms=("rms", "max"),
                                                      n_iter=("n_iter", "first"),
                                                      convergence_warnings=("convergence_warnings", "first"))
                     .reset_index())
        seeds = per_model[per_model.model.str.contains("seed=[0-4]$")]
        spread = (seeds.groupby("k")["mean_rms"].agg(["mean", "std", "min", "max"]).reset_index()
                  .assign(model="mlp h=(20,) alpha=0.001, seeds 0-4: mean / s.d. / min / max"))
        sens_table = per_model

    os.makedirs(a.outdir, exist_ok=True)
    tag = "" if a.tests == "ba" else f"_{a.tests}"
    folds.to_csv(os.path.join(a.outdir, f"ml_reconstruction_folds{tag}.csv"), index=False)
    summary.to_csv(os.path.join(a.outdir, f"ml_reconstruction_summary{tag}.csv"), index=False)
    if len(sens):
        sens.to_csv(os.path.join(a.outdir, "ml_sensitivity.csv"), index=False)
    if a.tests == "ba":
        plot_bars(summary, os.path.join(a.outdir, "ml_reconstruction.png"))

    pd.set_option("display.width", 160)
    fmt = lambda v: f"{v:.5f}"  # noqa: E731
    print("NOTE: the ensemble was aligned onto the reference mesh by nearest neighbour (not IDW).")
    print(f"{M} cases ({len(straight)} straight, {len(corner)} cornering), {X.shape[0]} points with y <= 0\n")
    print("Summary (RMS Cp error over all points):")
    print(summary[["test", "k", "model", "mean_rms", "worst_rms", "modes", "mlp_n_iter_median",
                   "convergence_warnings"]].to_string(index=False, float_format=fmt))
    if len(sens):
        print("\nMLP sensitivity, test (b):")
        print(sens_table.to_string(index=False, float_format=fmt))
        print("\nSpread across random_state 0-4, (20,) model, alpha 0.001 (mean RMS per seed):")
        print(spread[["k", "mean", "std", "min", "max"]].to_string(index=False, float_format=fmt))
    print(f"\nTotal run time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
