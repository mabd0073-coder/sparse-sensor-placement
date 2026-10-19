"""
read_vtu_field.py

Reads a baseline run's VTU/EnSight solution and extracts pressure at
every surface point, replacing rotate_and_interpolate.py's ascii-based
load_full_field() for this data format.

Requires: pip install pyvista --break-system-packages
"""

import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path


def load_vtm_field(vtm_path: Path) -> pd.DataFrame:
    """Load a VTK multi-block dataset and flatten every block's points
    and point-data arrays into one DataFrame.

    Returns a DataFrame with columns: block_index, x, y, z, plus one
    column per point-data array found (e.g. 'pressure', 'Absolute Pressure').
    """
    multiblock = pv.read(str(vtm_path))
    print(f"Loaded multiblock with {multiblock.n_blocks} blocks")

    rows = []
    for i in range(multiblock.n_blocks):
        block = multiblock[i]
        if block is None or block.n_points == 0:
            continue

        print(f"  Block {i}: {block.n_points} points, "
              f"arrays: {list(block.point_data.keys())}")

        points = block.points  # (n_points, 3)
        df_block = pd.DataFrame(points, columns=["x", "y", "z"])
        df_block["block_index"] = i

        for array_name in block.point_data.keys():
            values = np.asarray(block.point_data[array_name])
            if values.ndim == 1:
                df_block[array_name] = values
            # skip vector arrays (velocity etc.) for now — add explicitly if needed

        rows.append(df_block)

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    VTM_PATH = Path("solution_vtm.vtm")  # run from inside the out/ folder

    df = load_vtm_field(VTM_PATH)
    print(f"\nTotal points across all blocks: {len(df)}")
    print(f"Columns available: {list(df.columns)}")
    print(df.head())

    df.to_csv("full_field_extracted.csv", index=False)
    print("Saved to full_field_extracted.csv")