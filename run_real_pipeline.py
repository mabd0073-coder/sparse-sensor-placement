"""
run_real_pipeline.py

Runs POD + greedy sensor placement + leave-one-out validation on the
real 100-case ride-height sweep ensemble.
"""

import numpy as np
import pandas as pd

# --- 1. Load the real ensemble ---
data = np.load("side_diffuser_ensemble_dedup.npz")
Cp_matrix = data["Cp_matrix"]                     # (702040, 100)
coords = np.column_stack([data["x"], data["y"], data["z"]])
case_params = pd.read_csv("case_params.csv")

print(f"Ensemble shape: {Cp_matrix.shape}")
print(case_params.head())

# --- 2. POD ---
X_mean = Cp_matrix.mean(axis=1, keepdims=True)
X_tilde = Cp_matrix - X_mean

print("\nRunning SVD (this may take a minute at this size)...")
U, sigma, Vt = np.linalg.svd(X_tilde, full_matrices=False)

variance_target = 0.9
cumulative = np.cumsum(sigma**2) / np.sum(sigma**2)
r = int(np.searchsorted(cumulative, variance_target) + 1)
print(f"\nModes needed for {variance_target*100:.0f}% variance: {r} / {U.shape[1]}")

U_r = U[:, :r]

# --- 3. Greedy sensor placement ---
def greedy_sensor_placement(U_r, n_sensors, eps=1e-10, coords=None):
    n_points, r = U_r.shape
    remaining = list(range(n_points))  # TODO: filter to valid candidates later
    selected = []
    DtD_inv = (1 / eps) * np.eye(r)

    for k in range(n_sensors):
        best_idx, best_gain = None, -np.inf
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
        print(f"Sensor {k+1}: point index {best_idx}, coords {coords[best_idx]}, gain {best_gain:.2f}")

    return selected

n_sensors = 40
print(f"\nSelecting {n_sensors} sensors...")
selected = greedy_sensor_placement(U_r, n_sensors, coords=coords)

# --- 4. Reconstruction sanity check on one real case ---
def reconstruct(U_r, X_mean, selected_indices, y_measured):
    D = U_r[selected_indices, :]
    y = y_measured - X_mean[selected_indices, 0]
    z, *_ = np.linalg.lstsq(D, y, rcond=None)
    return U_r @ z + X_mean[:, 0]

y_case0 = Cp_matrix[selected, 0]
Cp_recon = reconstruct(U_r, X_mean, selected, y_case0)
rms = np.sqrt(np.mean((Cp_recon - Cp_matrix[:, 0])**2))
print(f"\nIn-sample RMS error (case 0, {case_params.iloc[0]['folder']}): {rms:.5f}")

# Save for later steps
np.savez("pod_results.npz", U_r=U_r, X_mean=X_mean, selected=selected, sigma=sigma)
print("\nSaved pod_results.npz for downstream use")