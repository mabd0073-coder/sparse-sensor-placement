"""
sensor_count_sweep.py

Sweeps the number of sensors (5, 10, 20, 30, 40) through leave-one-out
cross-validation on the real 100-case ride-height ensemble, to produce
an error-vs-sensor-count sensitivity plot for the report.

Run from the same folder as side_diffuser_ensemble_dedup.npz and
case_params.csv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------------------
# Load the real ensemble
# ---------------------------------------------------------------------------
data = np.load("side_diffuser_ensemble_dedup.npz")
Cp_matrix = data["Cp_matrix"]
case_params = pd.read_csv("case_params.csv")
n_cases = Cp_matrix.shape[1]

print(f"Ensemble shape: {Cp_matrix.shape}")

# ---------------------------------------------------------------------------
# Core functions (same as before)
# ---------------------------------------------------------------------------
def greedy_sensor_placement(U_r, n_sensors, eps=1e-10):
    n_points, r = U_r.shape
    remaining = list(range(n_points))
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

    return selected


def reconstruct(U_r, X_mean_train, selected_indices, y_measured):
    D = U_r[selected_indices, :]
    y = y_measured - X_mean_train[selected_indices, 0]
    z, *_ = np.linalg.lstsq(D, y, rcond=None)
    return U_r @ z + X_mean_train[:, 0]


def compute_pod(Cp_train, variance_target=0.9):
    X_mean_train = Cp_train.mean(axis=1, keepdims=True)
    X_tilde_train = Cp_train - X_mean_train
    U_train, sigma_train, _ = np.linalg.svd(X_tilde_train, full_matrices=False)
    cumulative = np.cumsum(sigma_train**2) / np.sum(sigma_train**2)
    r = int(np.searchsorted(cumulative, variance_target) + 1)
    r = min(r, U_train.shape[1])
    return U_train[:, :r], X_mean_train, r


def run_loo(Cp_matrix, n_sensors, variance_target=0.9, verbose=False):
    n_cases = Cp_matrix.shape[1]
    errors = []
    for test_idx in range(n_cases):
        train_cols = [i for i in range(n_cases) if i != test_idx]
        Cp_train = Cp_matrix[:, train_cols]
        Cp_test = Cp_matrix[:, test_idx]

        U_r, X_mean_train, r = compute_pod(Cp_train, variance_target)
        # n_sensors can't exceed number of modes usefully, but greedy still
        # works fine with more sensors than modes -- it just adds diminishing
        # value beyond r. We keep n_sensors as requested either way.
        selected = greedy_sensor_placement(U_r, n_sensors)

        y_test = Cp_test[selected]
        Cp_recon = reconstruct(U_r, X_mean_train, selected, y_test)
        err = np.sqrt(np.mean((Cp_recon - Cp_test) ** 2))
        errors.append(err)
        if verbose:
            print(f"    fold {test_idx+1}/{n_cases}: RMS error={err:.5f}")
    return np.array(errors)


# ---------------------------------------------------------------------------
# Sweep sensor counts
# ---------------------------------------------------------------------------
sensor_counts = [2, 5, 10, 20, 30, 40]
variance_target = 0.9

results = []
for n_sensors in sensor_counts:
    print(f"\nRunning LOO with {n_sensors} sensors...")
    t0 = time.time()
    errors = run_loo(Cp_matrix, n_sensors, variance_target)
    elapsed = time.time() - t0
    results.append({
        "n_sensors": n_sensors,
        "mean_rms": errors.mean(),
        "std_rms": errors.std(),
        "max_rms": errors.max(),
        "min_rms": errors.min(),
    })
    print(f"  n_sensors={n_sensors}: mean RMS={errors.mean():.5f}, "
          f"max RMS={errors.max():.5f}  (took {elapsed:.1f}s)")

results_df = pd.DataFrame(results)
results_df.to_csv("sensor_count_sweep_results.csv", index=False)
print("\n=== Sensor count sweep summary ===")
print(results_df.to_string(index=False))
print("\nSaved sensor_count_sweep_results.csv")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(results_df["n_sensors"], results_df["mean_rms"], "o-", label="Mean RMS error")
ax.fill_between(
    results_df["n_sensors"],
    results_df["mean_rms"] - results_df["std_rms"],
    results_df["mean_rms"] + results_df["std_rms"],
    alpha=0.2, label="+-1 std across 100 LOO folds",
)
ax.set_xlabel("Number of sensors")
ax.set_ylabel("Leave-one-out RMS Cp error")
ax.set_title("Reconstruction error vs sensor count (100-case ride-height sweep)")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig("sensor_count_sweep.png", dpi=200)
print("Saved sensor_count_sweep.png")