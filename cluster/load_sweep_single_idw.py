"""
load_sweep_ensemble_interpolated.py

Interpolates every case's side_diffuser.csv (different node count/
numbering per case) onto one fixed candidate point set, producing a
valid (N, M) Cp ensemble for POD.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SIDE_DIFFUSER_COLS = [
    "nodenumber", "x", "y", "z",
    "pressure", "dynamic_pressure", "total_pressure",
    "CpS", "velocity_magnitude",
]


class CfdDataError(ValueError):
    pass


def load_side_diffuser_case(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CfdDataError(f"File not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=0, names=SIDE_DIFFUSER_COLS, comment="#")
    if df.empty:
        raise CfdDataError(f"No rows parsed from {path}")
    if df[["x", "y", "z", "CpS"]].isna().any().any():
        raise CfdDataError(f"NaNs found in required columns in {path}")
    return df.reset_index(drop=True)


def parse_case_params(folder_name: str) -> dict:
    params = {}
    for match in re.finditer(r"([A-Z]+)(-?\d+\.?\d*)", folder_name):
        key, value = match.group(1), float(match.group(2))
        params[key] = value
    return params


def interpolate_case_onto_fixed_points(
    case_points: np.ndarray, case_cp: np.ndarray,
    fixed_points: np.ndarray, k: int = 4, eps: float = 1e-9,
) -> np.ndarray:
    """Inverse-distance weighted interpolation of case_cp (defined at
    case_points) onto fixed_points, using k nearest neighbours."""
    tree = cKDTree(case_points)
    distances, indices = tree.query(fixed_points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / (distances + eps)
    neighbor_values = case_cp[indices]
    return np.sum(weights * neighbor_values, axis=1) / np.sum(weights, axis=1)


def load_sweep_ensemble(
    sweep_dir: Path, fixed_points_path: Path, filename: str = "side_diffuser.csv",
):
    fixed_df = pd.read_csv(fixed_points_path)
    fixed_points = fixed_df[["x", "y", "z"]].to_numpy(dtype=float)
    print(f"Fixed candidate point set: {len(fixed_points)} points")

    case_folders = sorted(
        f for f in sweep_dir.iterdir()
        if f.is_dir() and "_" in f.name and (f / "out" / filename).exists()
    )
    print(f"Found {len(case_folders)} case folders with {filename}")

    cp_columns = []
    params_rows = []
    for folder in case_folders:
        csv_path = folder / "out" / filename
        try:
            df = load_side_diffuser_case(csv_path)
        except CfdDataError as e:
            print(f"  Skipping {folder.name}: {e}")
            continue

        case_points = df[["x", "y", "z"]].to_numpy(dtype=float)
        case_cp = df["CpS"].to_numpy(dtype=float)
        interp_cp = interpolate_case_onto_fixed_points(case_points, case_cp, fixed_points)

        cp_columns.append(interp_cp)
        params_rows.append({"folder": folder.name, **parse_case_params(folder.name)})
        print(f"  Interpolated {folder.name}: {len(df)} native nodes -> {len(fixed_points)} fixed points")

    Cp_matrix = np.column_stack(cp_columns)
    case_params = pd.DataFrame(params_rows)

    return fixed_df, Cp_matrix, case_params


if __name__ == "__main__":
    SWEEP_DIR = Path("/fs04/scratch2/he41/temp_runs/26_06/50.28")
    FIXED_POINTS = Path("all_points.csv")  # upload this to the cluster first, see below

    ref_df, Cp_matrix, case_params = load_sweep_ensemble(SWEEP_DIR, FIXED_POINTS)

    print(f"\nEnsemble shape: {Cp_matrix.shape} (fixed points x cases)")

    np.savez(
        "side_diffuser_ensemble.npz",
        Cp_matrix=Cp_matrix,
        x=ref_df["x"].to_numpy(),
        y=ref_df["y"].to_numpy(),
        z=ref_df["z"].to_numpy(),
    )
    case_params.to_csv("case_params.csv", index=False)
    print("\nSaved side_diffuser_ensemble.npz and case_params.csv")