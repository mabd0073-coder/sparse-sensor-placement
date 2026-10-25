"""
load_rrf_sweeps.py

Builds the RRF (rotating reference frame) training and test ensembles on the
cluster. Needs Python 3 with numpy, pandas and scipy only.

For every case folder in the given sweep directories it reads
<case>/out/side_diffuser.csv (or <case>/side_diffuser.csv), parses FRH, RRH,
R, S, CR, Y and V from the folder name, keeps only the left side (y <= 0),
and interpolates Cp onto the left-side points of the first complete training
case with the existing k=4 IDW function from master_load_sweep_ensemble.py.
Cases with no CSV (or an unreadable one) are skipped and listed.

Data on the right side only is mirrored to the left (y -> -y) first.

Replicates: the same design is often submitted to several sweep folders (the
folder name, e.g. 15_FRH52.94_RRH42.00_..., is identical). Only the first copy
found (in the order of --train) is kept, a test case that is also a training
case is dropped, and the difference between each pair of copies is written to
rrf_replicates.csv (a run-to-run repeatability check of the CFD). This keeps a
held-out case from also being in the training set.

Diagnostic: for each case, the fraction of target points whose 4 source
neighbours differ by more than --spread-threshold (default 0.3) in Cp. A high
fraction means the interpolation is averaging across a sharp feature or a
mesh mismatch.

Usage (defaults: train = 50.39 50.41, test = 50.40):
    python load_rrf_sweeps.py --base /fs04/scratch2/he41/temp_runs/26_06
    python load_rrf_sweeps.py --base <dir> --train 50.39 50.41 --test 50.40 --out-dir .

Outputs rrf_train.npz and rrf_test.npz with Cp_matrix (points x cases), x, y,
z (metres), case_names, sweep, FRH, RRH, R, S, CR, Y, V and idw_spread_frac.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from master_load_sweep_ensemble import (  # noqa: E402
    IDW_K, CfdDataError, idw_interpolate, load_side_diffuser_case, parse_case_params,
)

PARAM_KEYS = ["FRH", "RRH", "R", "S", "CR", "Y", "V"]
FILENAME = "side_diffuser.csv"


def find_csv(case_dir):
    for path in (case_dir / "out" / FILENAME, case_dir / FILENAME):
        if path.exists():
            return path
    return None


def list_cases(sweep_dirs):
    """(sweep name, case folder) for every case folder, sorted within each sweep."""
    cases = []
    for s_dir in sweep_dirs:
        if not s_dir.is_dir():
            print("WARNING: sweep directory not found: %s" % s_dir)
            continue
        for f in sorted(p for p in s_dir.iterdir() if p.is_dir() and "_" in p.name):
            cases.append((s_dir.name, f))
    return cases


def left_side(df):
    """Left-side (y <= 0) coordinates and Cp; right-only data is mirrored first."""
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    cp = df["CpS"].to_numpy(dtype=float)
    if coords[:, 1].min() > -0.05:  # data on the right side only
        coords = coords.copy()
        coords[:, 1] = -coords[:, 1]
    keep = coords[:, 1] <= 0
    return coords[keep], cp[keep]


def neighbour_spread_fraction(src_coords, src_cp, target_coords, threshold, k=IDW_K):
    """Fraction of target points whose k nearest source values span more than threshold."""
    _, idx = cKDTree(src_coords).query(target_coords, k=k)
    vals = src_cp[idx]
    return float(np.mean(vals.max(axis=1) - vals.min(axis=1) > threshold))


def load_cases(cases, label):
    """Read every case; returns list of (sweep, name, coords, cp) and the skipped list."""
    loaded, skipped = [], []
    for sweep, folder in cases:
        csv = find_csv(folder)
        if csv is None:
            skipped.append((sweep, folder.name, "no %s" % FILENAME))
            continue
        try:
            coords, cp = left_side(load_side_diffuser_case(csv))
        except (CfdDataError, ValueError, KeyError) as e:
            skipped.append((sweep, folder.name, str(e)))
            continue
        if len(coords) == 0:
            skipped.append((sweep, folder.name, "no points with y <= 0"))
            continue
        loaded.append((sweep, folder.name, coords, cp))
        print("  [%s] %s/%s: %d left-side nodes" % (label, sweep, folder.name, len(coords)))
    return loaded, skipped


def build_ensemble(loaded, target, threshold):
    columns, spread = [], []
    for sweep, name, coords, cp in loaded:
        if len(coords) == len(target) and np.allclose(coords, target, atol=1e-6):
            columns.append(cp)
        else:
            columns.append(idw_interpolate(coords, cp, target, k=IDW_K))
        spread.append(neighbour_spread_fraction(coords, cp, target, threshold))
    return np.column_stack(columns), np.array(spread)


def save(path, loaded, target, Cp_matrix, spread):
    params = [parse_case_params(name) for _, name, _, _ in loaded]
    np.savez_compressed(
        path,
        Cp_matrix=Cp_matrix,
        x=target[:, 0], y=target[:, 1], z=target[:, 2],
        case_names=np.array([name for _, name, _, _ in loaded]),
        sweep=np.array([sweep for sweep, _, _, _ in loaded]),
        idw_spread_frac=spread,
        **{k: np.array([p.get(k, np.nan) for p in params], dtype=float) for k in PARAM_KEYS}
    )
    print("Saved %s: %d points x %d cases" % (path, Cp_matrix.shape[0], Cp_matrix.shape[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="/fs04/scratch2/he41/temp_runs/26_06",
                    help="folder containing the sweep directories")
    ap.add_argument("--train", nargs="+", default=["50.39", "50.41"])
    ap.add_argument("--test", nargs="+", default=["50.40"])
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--spread-threshold", type=float, default=0.3)
    a = ap.parse_args()

    base = Path(a.base)
    sets = {}
    all_skipped = []
    for label, sweeps in (("train", a.train), ("test", a.test)):
        loaded, skipped = load_cases(list_cases([base / s for s in sweeps]), label)
        sets[label] = loaded
        all_skipped += [(label,) + s for s in skipped]

    if all_skipped:
        print("\nSkipped %d case(s):" % len(all_skipped))
        for label, sweep, name, why in all_skipped:
            print("  [%s] %s/%s: %s" % (label, sweep, name, why))
    if not sets["train"]:
        sys.exit("No complete training cases found; nothing saved.")

    # the same design can be in several sweep folders: keep one copy, record the pairs
    first, replicas = {}, []
    for label in ("train", "test"):
        kept = []
        for item in sets[label]:
            if item[1] in first:
                replicas.append((first[item[1]], item, label))
            else:
                first[item[1]] = item
                kept.append(item)
        sets[label] = kept
    if not sets["train"]:
        sys.exit("No complete training cases found; nothing saved.")

    _, ref_name, target, _ = sets["train"][0]
    print("\nTarget points: left side of %s (%d points), k=%d IDW" % (ref_name, len(target), IDW_K))

    os.makedirs(a.out_dir, exist_ok=True)
    if replicas:
        print("\nReplicate copies of the same design (second copy dropped from the ensembles):")
        rows = ["case,kept_sweep,replicate_sweep,replicate_in,rms_difference_Cp,max_abs_difference_Cp"]
        for kept, dup, label in replicas:
            c1 = build_ensemble([kept], target, a.spread_threshold)[0][:, 0]
            c2 = build_ensemble([dup], target, a.spread_threshold)[0][:, 0]
            d = c1 - c2
            rms, mx = float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max())
            print("  %s: %s vs %s: RMS difference %.4f, max %.3f" % (kept[1], kept[0], dup[0], rms, mx))
            rows.append("%s,%s,%s,%s,%.5f,%.4f" % (kept[1], kept[0], dup[0], label, rms, mx))
        with open(os.path.join(a.out_dir, "rrf_replicates.csv"), "w") as fh:
            fh.write("\n".join(rows) + "\n")
    for label in ("train", "test"):
        if not sets[label]:
            print("No %s cases found; rrf_%s.npz not written." % (label, label))
            continue
        Cp_matrix, spread = build_ensemble(sets[label], target, a.spread_threshold)
        print("\n%s: neighbour Cp spread > %g at %.2f%% of target points on average "
              "(worst case %.2f%%, %s)" % (label, a.spread_threshold, 100 * spread.mean(),
                                          100 * spread.max(), sets[label][int(spread.argmax())][1]))
        save(os.path.join(a.out_dir, "rrf_%s.npz" % label), sets[label], target, Cp_matrix, spread)


if __name__ == "__main__":
    main()
