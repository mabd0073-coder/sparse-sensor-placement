"""Extra validation numbers for the report (baseline, mode counts, extrapolation).

Prints, for a snapshot ensemble (N points x M cases):

1. Leave-one-out (LOO) RMS error of the POD + greedy-sensor + least-squares
   reconstruction, and the number of modes kept in each round.
2. The same LOO test for the "no sensors" baseline, which predicts the
   training-average field for every case. The ratio of 2 to 1 says how much
   the sensors actually help.
3. How much the cases differ: the largest spread between cases at any point
   and the median per-point standard deviation.
4. (Optional, needs case_params.csv with FRH and RRH columns in the same
   order as the ensemble columns) an extrapolation test: train on cases with
   both ride heights <= --edge mm and test on the rest, and the correlation of
   each case's mean Cp with FRH + RRH and with the pitch RRH - FRH.

Matching case_params rows to ensemble columns: if the .npz stores case names
(``case_names``), rows are matched by folder name. Otherwise the rows are taken
in file order; when the file has more rows than the ensemble has cases, pick
them with --rows (a pandas query, e.g. "sweep == 50.28"). The selected rows
must come from one sweep and vary only in FRH/RRH. --case-order names a CSV
(e.g. loo_results.csv) whose held_out_case column gives the folder of each
ensemble column; the selected rows must match it in order, and if it has an
rms_error column it is compared with this script's leave-one-out errors.

Usage:
    python experiments/validation_extras.py data/side_diffuser_ensemble_dedup.npz \
        --params data/case_params.csv --sensors 10
    python experiments/validation_extras.py data/side_diffuser_ensemble_dedup.npz \
        --params data/case_params.csv --rows "sweep == 50.28" --case-order data/loo_results.csv
"""
import sys

import argparse

import numpy as np
import pandas as pd


def compute_pod(X_train, variance_target=0.90):
    """Mean field, retained modes U_r and mode count r for a training matrix."""
    mean = X_train.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(X_train - mean, full_matrices=False)
    r = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), variance_target) + 1)
    r = min(r, U.shape[1])
    return mean, U[:, :r], r


def greedy(U_r, k, eps=1e-10):
    """Greedy D-optimal sensor selection (vectorised; same picks as analysis.py)."""
    n, r = U_r.shape
    sel, avail, D = [], np.ones(n, bool), np.eye(r) / eps
    for _ in range(k):
        g = 1.0 + ((U_r @ D) * U_r).sum(axis=1)
        g[~avail] = -np.inf
        b = int(np.argmax(g))
        sel.append(b)
        avail[b] = False
        w = U_r[b][:, None]
        Dw = D @ w
        D -= Dw @ Dw.T / (1.0 + (w.T @ Dw).item())
    return sel


def rebuild(U_r, mean, sel, x_true):
    """Least-squares reconstruction of a full field from readings at sel."""
    a, *_ = np.linalg.lstsq(U_r[sel], x_true[sel] - mean[sel, 0], rcond=None)
    return mean[:, 0] + U_r @ a


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_split(X, train_idx, test_idx, k, var):
    mean, U_r, r = compute_pod(X[:, train_idx], var)
    sel = greedy(U_r, k)
    pod = [rms(rebuild(U_r, mean, sel, X[:, j]), X[:, j]) for j in test_idx]
    base = [rms(mean[:, 0], X[:, j]) for j in test_idx]
    return pod, base, r


def select_params(cp, npz, M, rows=None, case_order=None):
    """Return the case_params rows matching the ensemble columns, in column order.

    Returns None (with a printed reason) when no safe match is possible.
    """
    if "case_names" in npz.files:
        names = [str(n) for n in npz["case_names"]]
        missing = [n for n in names if n not in set(cp["folder"])]
        if missing:
            sys.exit(f"[error] {len(missing)} ensemble case(s) not in params, e.g. {missing[0]}")
        print(f"\nParams matched by case name ({len(names)} cases)")
        return cp.set_index("folder").loc[names].reset_index()

    if rows is not None:
        cp = cp.query(rows).reset_index(drop=True)
        print(f"\nParams rows selected with --rows \"{rows}\": {len(cp)}")
    if len(cp) != M:
        hint = "" if rows is not None else "; use --rows to select the matching rows"
        print(f"\n[skip extrapolation] params have {len(cp)} rows, ensemble has {M} cases{hint}")
        return None

    if rows is not None:
        sweeps = cp["sweep"].unique() if "sweep" in cp else []
        if len(sweeps) != 1:
            sys.exit(f"[error] selected rows span several sweeps: {list(sweeps)}")
        fixed = [c for c in ("R", "S", "CR", "Y", "V") if c in cp and cp[c].nunique() > 1]
        if fixed:
            sys.exit(f"[error] selected rows are not ride-height-only cases; {fixed} vary")
        print(f"  all from sweep {sweeps[0]:g}; only FRH/RRH vary")

    if case_order is not None:
        order = pd.read_csv(case_order)["held_out_case"].tolist()
        if order != cp["folder"].tolist():
            bad = next(i for i, (a, b) in enumerate(zip(order, cp["folder"])) if a != b) \
                if len(order) == len(cp) else None
            sys.exit(f"[error] params rows do not match the column order in {case_order}"
                     + (f" (first mismatch at column {bad})" if bad is not None else
                        f" ({len(order)} vs {len(cp)} entries)"))
        print(f"  folder order matches {case_order} for all {M} columns")
    return cp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble", help=".npz file with a Cp_matrix array (N points x M cases)")
    p.add_argument("--key", default="Cp_matrix")
    p.add_argument("--params", default=None, help="case_params.csv with FRH and RRH columns")
    p.add_argument("--sensors", type=int, default=10)
    p.add_argument("--variance", type=float, default=0.90)
    p.add_argument("--edge", type=float, default=50.0, help="extrapolation split [mm]")
    p.add_argument("--rows", default=None,
                   help='pandas query selecting the params rows, e.g. "sweep == 50.28"')
    p.add_argument("--case-order", default=None,
                   help="CSV with a held_out_case column giving the folder of each ensemble column")
    a = p.parse_args()

    npz = np.load(a.ensemble)
    X = npz[a.key]
    N, M = X.shape
    print(f"Ensemble {a.ensemble}: {N} points x {M} cases")

    spread = X.max(axis=1) - X.min(axis=1)
    print(f"Case-to-case spread: largest at any point {spread.max():.4f}; "
          f"median per-point s.d. {np.median(X.std(axis=1)):.4f}")

    pod, base, modes = [], [], []
    for j in range(M):
        train = [i for i in range(M) if i != j]
        e_pod, e_base, r = test_split(X, train, [j], a.sensors, a.variance)
        pod += e_pod
        base += e_base
        modes.append(r)
    pod, base = np.array(pod), np.array(base)
    print(f"\nLeave-one-out, {a.sensors} sensors, {a.variance:.0%} variance target")
    print(f"  POD reconstruction RMS: mean {pod.mean():.5f}  s.d. {pod.std():.5f}  "
          f"min {pod.min():.5f}  max {pod.max():.5f}")
    print(f"  mean-field baseline RMS: mean {base.mean():.5f}  s.d. {base.std():.5f}  max {base.max():.5f}")
    print(f"  improvement factor (baseline / POD): {base.mean() / pod.mean():.1f}")
    vals, counts = np.unique(modes, return_counts=True)
    print("  modes kept per round: " + ", ".join(f"{v} mode(s) in {c} rounds" for v, c in zip(vals, counts)))

    if a.case_order:
        ref = pd.read_csv(a.case_order)
        if "rms_error" in ref and len(ref) == M:
            diff = np.abs(ref["rms_error"].to_numpy(float) - pod)
            print(f"  per-column LOO RMS vs {a.case_order}: max abs difference {diff.max():.2e}")

    if a.params:
        cp = select_params(pd.read_csv(a.params), npz, M, a.rows, a.case_order)
        if cp is None:
            return
        frh, rrh = cp["FRH"].to_numpy(float), cp["RRH"].to_numpy(float)
        mean_cp = X.mean(axis=0)
        print(f"\nCorrelation of each case's mean Cp with FRH + RRH: {np.corrcoef(frh + rrh, mean_cp)[0, 1]:.3f}; "
              f"with pitch RRH - FRH: {np.corrcoef(rrh - frh, mean_cp)[0, 1]:.3f}")
        inside = (frh <= a.edge) & (rrh <= a.edge)
        tr, te = np.where(inside)[0], np.where(~inside)[0]
        if len(te) == 0 or len(tr) < 3:
            print("[skip extrapolation] split leaves too few cases")
            return
        e_pod, e_base, r = test_split(X, tr, te, a.sensors, a.variance)
        print(f"\nExtrapolation: train on {len(tr)} cases with FRH and RRH <= {a.edge:g} mm, "
              f"test on the other {len(te)} ({r} mode(s) kept)")
        print(f"  POD RMS mean {np.mean(e_pod):.5f} (max {np.max(e_pod):.5f}); "
              f"mean-field baseline RMS mean {np.mean(e_base):.5f}")


if __name__ == "__main__":
    main()
