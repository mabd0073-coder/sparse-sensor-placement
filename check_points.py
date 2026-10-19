import pandas as pd
import numpy as np

# Load the new file with exclusion flags
new_points = pd.read_csv("all_points_with_exclusion.csv")
coords = new_points[["x", "y", "z"]].to_numpy()
excluded = new_points["excluded"].astype(bool).to_numpy()

# Reproduce the exact same dedup logic used to build side_diffuser_ensemble_dedup.npz
decimals = 6
rounded = np.round(coords, decimals=decimals)
_, unique_indices = np.unique(rounded, axis=0, return_index=True)
unique_indices = np.sort(unique_indices)

coords_dedup = coords[unique_indices]
excluded_dedup = excluded[unique_indices]

print(f"After dedup: {len(coords_dedup)} points")

# Sanity check: does this match your existing dedup'd ensemble exactly?
data = np.load("side_diffuser_ensemble_dedup.npz")
old_coords = np.column_stack([data["x"], data["y"], data["z"]])

print("Same count as ensemble?", len(coords_dedup) == len(old_coords))
print("Coordinates match exactly?", np.allclose(coords_dedup, old_coords))

if np.allclose(coords_dedup, old_coords):
    np.savez("exclusion_mask.npz", excluded=excluded_dedup)
    print(f"\nSaved exclusion_mask.npz: {excluded_dedup.sum()} / {len(excluded_dedup)} points excluded")
else:
    print("\nWARNING: coordinates don't line up — dedup ordering may differ, need index matching instead")