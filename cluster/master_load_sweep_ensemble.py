"""
master_load_sweep_ensemble.py

Loads side_diffuser.csv exports from multiple ride-height sweep directories.
Features:
1. Searches all sweep directories and parses case parameters.
2. Automatically detects and mirrors half-car (left-hand) symmetric sweeps.
3. Automatically identifies the highest-density mesh to use as the target reference.
4. Spatially maps lower-density meshes onto the high-density target using KDTree.
"""

from pathlib import Path
import re
from typing import Union, Sequence, Tuple, List
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SIDE_DIFFUSER_COLS = [
    "nodenumber", "x", "y", "z",
    "pressure", "dynamic_pressure", "total_pressure",
    "CpS", "velocity_magnitude",
]

class CfdDataError(ValueError):
    """Custom exception raised when CFD output files fail validation checks."""
    pass

def load_side_diffuser_case(path: Path) -> pd.DataFrame:
    """Load and validate a single CFD side_diffuser.csv data file."""
    if not path.exists():
        raise CfdDataError(f"File not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=0, names=SIDE_DIFFUSER_COLS, comment="#")
    if df.empty:
        raise CfdDataError(f"No rows parsed from {path}")
    if df[["x", "y", "z", "CpS"]].isna().any().any():
        raise CfdDataError(f"NaNs found in required columns in {path}")
    
    df = df.reset_index(drop=True)
    
    # Convert mm coordinates to meters if detected
    if np.max(np.abs(df["x"].values)) > 10.0:
        df["x"] /= 1000.0
        df["y"] /= 1000.0
        df["z"] /= 1000.0

    return df

def parse_case_params(folder_name: str) -> dict:
    """Extract key-value parameter pairs encoded in a case folder name."""
    params = {}
    for match in re.finditer(r"([A-Z]+)(-?\d+\.?\d*)", folder_name):
        key, value = match.group(1), float(match.group(2))
        params[key] = value
    return params

def prepare_full_car_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Checks if a case is left-hand (half-car) symmetric or full-car.
    If half-car, mirrors coordinates across y = 0 and duplicates Cp values.
    """
    coords = df[["x", "y", "z"]].to_numpy()
    cp = df["CpS"].to_numpy()

    # If minimum Y is close to or above 0, it's a left-hand (half-car) mesh
    is_half_car = df["y"].min() > -0.05

    if is_half_car:
        # Create right-hand mirrored coordinates (y -> -y)
        coords_rh = coords.copy()
        coords_rh[:, 1] = -coords_rh[:, 1]

        coords_full = np.vstack([coords, coords_rh])
        cp_full = np.concatenate([cp, cp])
        return coords_full, cp_full, True

    return coords, cp, False

def load_sweep_ensemble(
    sweep_dirs: Union[Sequence[Path], Path], 
    filename: str = "side_diffuser.csv"
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load, mirror, align, and concatenate CFD cases.
    Automatically sets the largest spatial grid as the reference mesh.
    """
    if isinstance(sweep_dirs, (Path, str)):
        sweep_dirs = [Path(sweep_dirs)]
    else:
        sweep_dirs = [Path(d) for d in sweep_dirs]

    case_folders = []
    for s_dir in sweep_dirs:
        if not s_dir.exists():
            print(f"Warning: Sweep directory does not exist: {s_dir}")
            continue
        folders = sorted(
            f for f in s_dir.iterdir()
            if f.is_dir() and "_" in f.name and (f / "out" / filename).exists()
        )
        for f in folders:
            case_folders.append((s_dir.name, f))

    if not case_folders:
        raise CfdDataError(f"No case folders with {filename} found under provided directories.")

    print(f"Found {len(case_folders)} total case folders across {len(sweep_dirs)} sweep directories.")

    raw_cases: List[Tuple[np.ndarray, np.ndarray]] = []
    params_rows = []

    # 1. Load all data into memory and apply mirroring if needed
    for sweep_name, folder in case_folders:
        csv_path = folder / "out" / filename
        try:
            df = load_side_diffuser_case(csv_path)
        except CfdDataError as e:
            print(f"  Skipping [{sweep_name}] {folder.name}: {e}")
            continue

        coords_full, cp_full, was_mirrored = prepare_full_car_data(df)
        raw_cases.append((coords_full, cp_full))

        status_str = f"Mirrored -> {len(coords_full)} nodes" if was_mirrored else f"Full-car -> {len(coords_full)} nodes"
        print(f"  Loaded [{sweep_name}] {folder.name}: {status_str}")

        params_rows.append({
            "sweep": sweep_name,
            "folder": folder.name,
            "is_mirrored": was_mirrored,
            "node_count": len(coords_full),
            **parse_case_params(folder.name)
        })

    if not raw_cases:
        raise CfdDataError("No valid cases were loaded.")

    # 2. Select target reference mesh coordinates (Find the case with maximum node count)
    target_idx = np.argmax([p["node_count"] for p in params_rows])
    target_coords, _ = raw_cases[target_idx]
    
    print(f"\nUsing case [{params_rows[target_idx]['sweep']}/{params_rows[target_idx]['folder']}] ")
    print(f"as spatial reference grid ({len(target_coords):,} target nodes).")

    # 3. Spatial alignment via KDTree interpolation mapping coarse cases to high-density target
    aligned_cp_list = []
    for i, (coords, cp) in enumerate(raw_cases):
        if len(coords) == len(target_coords) and np.allclose(coords, target_coords, atol=1e-5):
            aligned_cp_list.append(cp)
        else:
            # Create KDTree on the source (coarse) mesh, and query using the target (fine) mesh
            tree = cKDTree(coords)
            _, nearest_indices = tree.query(target_coords)
            cp_aligned = cp[nearest_indices]
            aligned_cp_list.append(cp_aligned)

    Cp_matrix = np.column_stack(aligned_cp_list)
    case_params = pd.DataFrame(params_rows)

    ref_df = pd.DataFrame({
        "nodenumber": np.arange(len(target_coords)),
        "x": target_coords[:, 0],
        "y": target_coords[:, 1],
        "z": target_coords[:, 2],
    })

    return ref_df, Cp_matrix, case_params

if __name__ == "__main__":
    SWEEP_DIRS = [
        Path("/fs04/scratch2/he41/temp_runs/26_06/50.28"),
        Path("/fs04/scratch2/he41/temp_runs/26_06/50.34"),
    ]

    ref_df, Cp_matrix, case_params = load_sweep_ensemble(SWEEP_DIRS)

    print(f"\nEnsemble shape: {Cp_matrix.shape} (nodes x cases)")
    
    np.savez_compressed(
        "side_diffuser_ensemble_highres.npz",
        Cp_matrix=Cp_matrix,
        x=ref_df["x"].to_numpy(),
        y=ref_df["y"].to_numpy(),
        z=ref_df["z"].to_numpy(),
        nodenumber=ref_df["nodenumber"].to_numpy(),
    )
    case_params.to_csv("case_params_highres.csv", index=False)
    print("\nSaved side_diffuser_ensemble_highres.npz and case_params_highres.csv")
