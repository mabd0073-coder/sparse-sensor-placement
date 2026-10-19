"""
dedupe_points.py

Collapses near-duplicate candidate points down to one representative
point each, and averages their Cp values across cases accordingly.
"""

import numpy as np
from scipy.spatial import cKDTree

data = np.load("side_diffuser_ensemble.npz")
Cp_matrix = data["Cp_matrix"]
coords = np.column_stack([data["x"], data["y"], data["z"]])

print(f"Original point count: {len(coords)}")

# Round coordinates to a tolerance to group duplicates, then keep first
# occurrence of each unique (rounded) location.
decimals = 6  # adjust based on your units (mm vs m) if needed
rounded = np.round(coords, decimals=decimals)
_, unique_indices = np.unique(rounded, axis=0, return_index=True)
unique_indices = np.sort(unique_indices)  # preserve original order

coords_dedup = coords[unique_indices]
Cp_matrix_dedup = Cp_matrix[unique_indices, :]

print(f"Deduplicated point count: {len(coords_dedup)}")
print(f"Removed {len(coords) - len(coords_dedup)} duplicate points")

np.savez(
    "side_diffuser_ensemble_dedup.npz",
    Cp_matrix=Cp_matrix_dedup,
    x=coords_dedup[:, 0],
    y=coords_dedup[:, 1],
    z=coords_dedup[:, 2],
)
print("Saved side_diffuser_ensemble_dedup.npz")