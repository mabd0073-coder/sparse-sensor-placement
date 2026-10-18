"""
sensor_pipeline.py

Sparse sensor placement for the side diffuser, using real multi-attitude
aeromap CFD cases instead of a single noisy case.
"""

import os
import glob
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

AIR_DENSITY_KG_M3 = 1.225
FREESTREAM_VELOCITY_MS = 19.44  # override per case if speed varies

CFD_COLUMNS = [
    "nodenumber", "x", "y", "z",
    "velocity_magnitude", "total_pressure", "dynamic_pressure", "pressure",
    "z2", "y2", "x2",
]


class CfdDataError(ValueError):
    """Raised when a CFD export is missing, malformed, or inconsistent
    with the rest of the ensemble."""


def compute_cp(pressure: np.ndarray, rho: float = AIR_DENSITY_KG_M3,
               v_inf: float = FREESTREAM_VELOCITY_MS) -> np.ndarray:
    """Cp = p / (0.5 * rho * v_inf^2)."""
    return pressure / (0.5 * rho * v_inf**2)


def load_cfd_case(path: Path, v_inf: float = FREESTREAM_VELOCITY_MS) -> pd.DataFrame:
    """Load a single Fluent ASCII point export and compute CpS.

    Parameters
    ----------
    path : path to the whitespace-delimited export file
    v_inf : freestream velocity for this case [m/s]

    Raises
    ------
    CfdDataError if the file is missing, empty, or has NaNs in required columns.
    """
    if not path.exists():
        raise CfdDataError(f"CFD export not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=0, names=CFD_COLUMNS, comment="#")

    if df.empty:
        raise CfdDataError(f"No rows parsed from {path}")
    if df[["x", "y", "z", "pressure"]].isna().any().any():
        raise CfdDataError(f"NaNs found in required columns in {path}")

    # IMPORTANT: do not sort by z here. Sorting independently per case is
    # how you silently break node correspondence across cases if any two
    # nodes share a z value in a different x/y order. Node order must be
    # guaranteed identical across all cases by the upstream export/rotation
    # pipeline (see project pseudocode), not fixed up here.
    df["CpS"] = compute_cp(df["pressure"].to_numpy(), v_inf=v_inf)
    return df.reset_index(drop=True)


def discover_case_files(data_dir: Path, pattern: str = "side_diffuser_case*.txt") -> List[Path]:
    """Find all case files in data_dir matching pattern, sorted for
    reproducible ordering."""
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise CfdDataError(f"No files matching '{pattern}' found in {data_dir}")
    return files


def load_case_ensemble(
    case_files: Sequence[Path],
    v_inf_per_case: Optional[Sequence[float]] = None,
    case_labels: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load multiple aeromap cases sharing the same node ordering into a
    single N x M pressure-coefficient matrix, ready for POD.

    Parameters
    ----------
    case_files : ordered list of per-case export file paths
    v_inf_per_case : optional per-case freestream speed (defaults to the
        module constant for every case)
    case_labels : optional human-readable label per case (e.g. attitude
        description), used only for reporting/plots — defaults to filenames

    Returns
    -------
    reference_df : DataFrame from the first case (coordinates etc, for
        plotting and sensor coordinate lookup)
    Cp_matrix : (N, M) array — one column per case
    labels : list of M case labels

    Raises
    ------
    CfdDataError if any two cases have a different node count.
    """
    v_infs = v_inf_per_case or [FREESTREAM_VELOCITY_MS] * len(case_files)
    if len(v_infs) != len(case_files):
        raise CfdDataError("v_inf_per_case length must match case_files length")

    dfs = [load_cfd_case(p, v_inf=v) for p, v in zip(case_files, v_infs)]

    n_nodes = {len(d) for d in dfs}
    if len(n_nodes) != 1:
        counts = {p.name: len(d) for p, d in zip(case_files, dfs)}
        raise CfdDataError(f"Mismatched node counts across cases: {counts}")

    Cp_matrix = np.column_stack([d["CpS"].to_numpy() for d in dfs])
    labels = list(case_labels) if case_labels else [p.stem for p in case_files]

    return dfs[0], Cp_matrix, labels


# ---------------------------------------------------------------------------
# 2. POD / SVD
# ---------------------------------------------------------------------------

def compute_pod(Cp_matrix: np.ndarray, variance_target: float = 0.9):
    """Perform POD (mean-centred SVD) on an (N, M) Cp ensemble.

    Returns
    -------
    U_r : (N, r) truncated spatial modes
    X_mean : (N, 1) per-node mean across cases
    r : number of modes retained to hit variance_target
    sigma : full singular value spectrum (for diagnostics/plots)
    """
    X_mean = Cp_matrix.mean(axis=1, keepdims=True)
    X_tilde = Cp_matrix - X_mean

    U, sigma, _Vt = np.linalg.svd(X_tilde, full_matrices=False)

    cumulative = np.cumsum(sigma**2) / np.sum(sigma**2)
    r = int(np.searchsorted(cumulative, variance_target) + 1)
    r = min(r, U.shape[1])  # guard: can't exceed available modes

    print(f"Modes needed for {variance_target*100:.0f}% variance: {r} / {U.shape[1]}")
    return U[:, :r], X_mean, r, sigma


# ---------------------------------------------------------------------------
# 3. Greedy sensor placement
# ---------------------------------------------------------------------------

def greedy_sensor_placement(U_r: np.ndarray, n_sensors: int,
                             candidate_indices: Optional[Sequence[int]] = None,
                             eps: float = 1e-10,
                             coords: Optional[np.ndarray] = None,
                             verbose: bool = True) -> List[int]:
    """Greedily select sensor node indices that maximise D-optimality
    (Woodbury/Sherman-Morrison update), restricted to physically valid
    candidate nodes.

    Parameters
    ----------
    U_r : (N, r) truncated POD modes
    n_sensors : number of sensors to place
    candidate_indices : node indices sensors are allowed to occupy
        (e.g. after excluding zones too close to brackets/internals).
        Defaults to every node if not supplied — NOTE: for a real build
        you should always pass a filtered candidate list.
    coords : (N, 3) array of node coordinates, used only for the printout
    """
    n_points, r = U_r.shape
    remaining = list(candidate_indices) if candidate_indices is not None else list(range(n_points))
    if len(remaining) < n_sensors:
        raise ValueError(f"Only {len(remaining)} candidate points available for {n_sensors} sensors")

    selected: List[int] = []
    DtD_inv = (1 / eps) * np.eye(r)

    for k in range(n_sensors):
        best_idx = None
        best_gain = -np.inf
        for i in remaining:
            w = U_r[i, :]
            gain = 1.0 + float(w @ DtD_inv @ w)
            if gain > best_gain:
                best_gain, best_idx = gain, i

        selected.append(best_idx)
        remaining.remove(best_idx)

        w = U_r[best_idx, :].reshape(-1, 1)
        Dw = DtD_inv @ w
        denom = 1.0 + (w.T @ Dw).item()
        DtD_inv -= (Dw @ Dw.T) / denom

        if verbose:
            loc = f", coords {coords[best_idx]}" if coords is not None else ""
            print(f"Sensor {k+1}: point index {best_idx}{loc}, gain {best_gain:.4f}")

    return selected


# ---------------------------------------------------------------------------
# 4. Reconstruction
# ---------------------------------------------------------------------------

def reconstruct(U_r: np.ndarray, X_mean: np.ndarray,
                 selected_indices: Sequence[int], y_measured: np.ndarray) -> np.ndarray:
    """Reconstruct the full Cp field from sparse sensor readings.

    y_measured : Cp values at the sensor locations, shape (p,)
    """
    D = U_r[selected_indices, :]
    y = y_measured - X_mean[selected_indices, 0]
    z, *_ = np.linalg.lstsq(D, y, rcond=None)
    return U_r @ z + X_mean[:, 0]


def rms_error(Cp_recon: np.ndarray, Cp_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((Cp_recon - Cp_true) ** 2)))


# ---------------------------------------------------------------------------
# 5. Leave-one-out cross-validation
#    This is the real validation step, now that we have genuine multiple
#    physical cases instead of noisy copies of one case.
# ---------------------------------------------------------------------------

def leave_one_out_validation(
    Cp_matrix: np.ndarray,
    n_sensors: int,
    candidate_indices: Optional[Sequence[int]] = None,
    variance_target: float = 0.9,
    coords: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """For each case, train POD + sensor selection on every OTHER case,
    then reconstruct the held-out case from sensor readings and measure
    error. Returns one row of results per held-out case.
    """
    n_cases = Cp_matrix.shape[1]
    results = []

    for test_idx in range(n_cases):
        train_cols = [i for i in range(n_cases) if i != test_idx]
        Cp_train = Cp_matrix[:, train_cols]
        Cp_test = Cp_matrix[:, test_idx]

        U_r, X_mean, r, _sigma = compute_pod(Cp_train, variance_target)
        selected = greedy_sensor_placement(
            U_r, n_sensors, candidate_indices=candidate_indices,
            coords=coords, verbose=False,
        )

        y_test = Cp_test[selected]
        Cp_recon = reconstruct(U_r, X_mean, selected, y_test)
        err = rms_error(Cp_recon, Cp_test)

        results.append({"held_out_case": test_idx, "n_modes": r, "rms_error": err})
        print(f"[LOO] held-out case {test_idx}: modes={r}, RMS error={err:.5f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = Path(".")                       # folder containing case files
    CASE_PATTERN = "side_diffuser_case*.txt"   # adjust to match Mark's naming
    N_SENSORS = 40
    VARIANCE_TARGET = 0.9

    # --- load real cases ---
    case_files = discover_case_files(DATA_DIR, CASE_PATTERN)
    print(f"Found {len(case_files)} case files")

    ref_df, Cp_matrix, labels = load_case_ensemble(case_files)
    coords = ref_df[["x", "y", "z"]].values
    print(f"Ensemble shape: {Cp_matrix.shape} (nodes x cases)")

    # --- candidate sensor locations ---
    # TODO: replace with the real exclusion-zone filtered list once the
    # STL/CDK-tree pipeline is wired in. Placeholder: every node allowed.
    candidate_indices = list(range(Cp_matrix.shape[0]))

    # --- POD on the full ensemble ---
    U_r, X_mean, r, sigma = compute_pod(Cp_matrix, VARIANCE_TARGET)

    # --- sensor placement ---
    selected = greedy_sensor_placement(
        U_r, N_SENSORS, candidate_indices=candidate_indices, coords=coords,
    )
    sensors_df = ref_df.iloc[selected]
    print("\nSelected sensors:\n", sensors_df[["nodenumber", "x", "y", "z"]])

    # --- in-sample reconstruction sanity check (case 0) ---
    y_case0 = Cp_matrix[selected, 0]
    Cp_recon_case0 = reconstruct(U_r, X_mean, selected, y_case0)
    print(f"\nIn-sample RMS error (case 0): {rms_error(Cp_recon_case0, Cp_matrix[:, 0]):.5f}")

    # --- the real validation: leave-one-out across cases ---
    print("\nRunning leave-one-out cross-validation...")
    loo_results = leave_one_out_validation(
        Cp_matrix, N_SENSORS, candidate_indices=candidate_indices,
        variance_target=VARIANCE_TARGET, coords=coords,
    )
    print("\nLOO summary:")
    print(loo_results.describe())

    # --- plot RMS error vs sensor count (on held-out data) ---
    sensor_counts = [5, 10, 20, 30, 40]
    sweep_errors = []
    for n in sensor_counts:
        loo = leave_one_out_validation(
            Cp_matrix, n, candidate_indices=candidate_indices,
            variance_target=VARIANCE_TARGET,
        )
        sweep_errors.append(loo["rms_error"].mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sensor_counts, sweep_errors, "o-")
    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Mean LOO RMS Cp error")
    ax.set_title("Reconstruction error vs sensor count (leave-one-out)")
    fig.tight_layout()
    fig.savefig("sensor_count_sweep.png", dpi=200)
    print("Saved plot to sensor_count_sweep.png")