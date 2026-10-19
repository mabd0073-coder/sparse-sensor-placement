"""
diagnostics.py

Checks whether the real ensemble actually varies across cases, and
whether the fixed candidate points contain near-duplicates.
"""

import numpy as np
from scipy.spatial import cKDTree

# --- Load the same data ---
data = np.load("side_diffuser_ensemble.npz")
Cp_matrix = data["Cp_matrix"]
coords = np.column_stack([data["x"], data["y"], data["z"]])

# --- Check 1: does Cp_matrix actually vary across cases? ---
print("Std across cases per point (first 5 points):")
print(Cp_matrix[:5, :].std(axis=1))

print("\nOverall mean variance across all points:")
print(Cp_matrix.var(axis=1).mean())

print("\nAre case 0 and case 1 nearly identical?")
print(np.allclose(Cp_matrix[:, 0], Cp_matrix[:, 1], atol=0.01))

print("\nMax absolute difference between case 0 and case 1:")
print(np.max(np.abs(Cp_matrix[:, 0] - Cp_matrix[:, 1])))

# --- Check 2: are there near-duplicate candidate points? ---
print("\nBuilding KD-tree to check for near-duplicate points...")
tree = cKDTree(coords)
pairs = tree.query_pairs(r=1e-6)
print(f"Near-duplicate point pairs (within 1e-6 units): {len(pairs)}")

# Also check a slightly larger tolerance, in case units are mm not m
pairs_larger = tree.query_pairs(r=1e-3)
print(f"Near-duplicate point pairs (within 1e-3 units): {len(pairs_larger)}")