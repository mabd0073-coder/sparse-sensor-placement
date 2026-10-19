"""
baseline_comparison_and_noise.py

Two additions to the sensor-placement study:
1. Compares the greedy D-optimal placement against two baselines:
   - QR decomposition with column pivoting (a standard alternative in
     the sparse-sensor-placement literature)
   - Random sensor placement (naive baseline)
   All three are evaluated via the same leave-one-out RMS error metric.

2. Noise robustness: perturbs sensor readings with Gaussian noise at
   several levels and measures how reconstruction error degrades, and
   how that degradation changes with sensor count (more sensors should
   act as a noise filter via the least-squares fit).

Run from the same folder as side_diffuser_ensemble_dedup.npz and
case_params.csv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import qr
import time

# ---------------------------------------------------------------------------
# Load ensemble
# ---------------------------------------------------------------------------
data = np.load("side_diffuser_ensemble_dedup.npz")
Cp_matrix = data["Cp_matrix"]
n_points, n_cases = Cp_matrix.shape
print(f"Ensemble shape: {Cp_matrix.shape}")

# ---------------------------------------------------------------------------
# Shared POD / reconstruction machinery
# ---------------------------------------------------------------------------
def compute_pod(Cp_train, variance_target=0.9):
    X_mean_train = Cp_train.mean(axis=1, keepdims=True)
    X_tilde_train = Cp_train - X_mean_train
    U_train, sigma_train, _ = np.linalg.svd(X_tilde_train, full_matrices=False)
    cumulative = np.cumsum(sigma_train**2) / np.sum(sigma_train**2)
    r = int(np.searchsorted(cumulative, variance_target) + 1)
    r = min(r, U_train.shape[1])
    return U_train[:, :r], X_mean_train, r


def reconstruct(U_r, X_mean_train, selected_indices, y_measured):
    D = U_r[selected_indices, :]
    y = y_measured - X_mean_train[selected_indices, 0]
    z, *_ = np.linalg.lstsq(D, y, rcond=None)
    return U_r @ z + X_mean_train[:, 0]


# ---------------------------------------------------------------------------
# Placement algorithm 1: your existing greedy D-optimal method
# ---------------------------------------------------------------------------
def greedy_sensor_placement(U_r, n_sensors, eps=1e-10):
    n_pts, r = U_r.shape
    remaining = list(range(n_pts))
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


# ---------------------------------------------------------------------------
# Placement algorithm 2: QR with column pivoting
# Standard alternative from the sparse-sensor-placement literature
# (Manohar et al.). Pivoting on U_r^T ranks points by how much they
# reduce conditioning of the sensing matrix, one column at a time.
# ---------------------------------------------------------------------------
def qr_pivot_sensor_placement(U_r, n_sensors):
    n_pts, r = U_r.shape
    # QR with pivoting on the transpose: pivots are candidate sensor indices
    _, _, pivots = qr(U_r.T, pivoting=True)
    return list(pivots[:n_sensors])


# ---------------------------------------------------------------------------
# Placement algorithm 3: random baseline
# ---------------------------------------------------------------------------
def random_sensor_placement(n_points, n_sensors, seed=42):
    rng = np.random.default_rng(seed)
    return list(rng.choice(n_points, size=n_sensors, replace=False))


# ---------------------------------------------------------------------------
# Leave-one-out evaluation for a given placement function
# ---------------------------------------------------------------------------
def run_loo(Cp_matrix, n_sensors, placement_fn, variance_target=0.9):
    n_cases = Cp_matrix.shape[1]
    errors = []
    for test_idx in range(n_cases):
        train_cols = [i for i in range(n_cases) if i != test_idx]
        Cp_train = Cp_matrix[:, train_cols]
        Cp_test = Cp_matrix[:, test_idx]

        U_r, X_mean_train, r = compute_pod(Cp_train, variance_target)
        selected = placement_fn(U_r, n_sensors)

        y_test = Cp_test[selected]
        Cp_recon = reconstruct(U_r, X_mean_train, selected, y_test)
        err = np.sqrt(np.mean((Cp_recon - Cp_test) ** 2))
        errors.append(err)
    return np.array(errors)


# ---------------------------------------------------------------------------
# PART 1: Compare placement algorithms
# ---------------------------------------------------------------------------
n_sensors = 10
variance_target = 0.9

print(f"\n=== Comparing placement algorithms ({n_sensors} sensors) ===")

algorithms = {
    "Greedy (D-optimal)": greedy_sensor_placement,
    "QR pivoting": qr_pivot_sensor_placement,
    "Random": lambda U_r, n: random_sensor_placement(U_r.shape[0], n),
}

comparison_results = []
for name, fn in algorithms.items():
    print(f"\nRunning LOO for: {name}...")
    t0 = time.time()
    errors = run_loo(Cp_matrix, n_sensors, fn, variance_target)
    elapsed = time.time() - t0
    comparison_results.append({
        "algorithm": name,
        "mean_rms": errors.mean(),
        "std_rms": errors.std(),
        "max_rms": errors.max(),
    })
    print(f"  {name}: mean RMS={errors.mean():.5f}, max RMS={errors.max():.5f}  "
          f"(took {elapsed:.1f}s)")

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv("placement_algorithm_comparison.csv", index=False)
print("\n=== Placement algorithm comparison ===")
print(comparison_df.to_string(index=False))
print("Saved placement_algorithm_comparison.csv")

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(comparison_df["algorithm"], comparison_df["mean_rms"],
       yerr=comparison_df["std_rms"], capsize=5)
ax.set_ylabel("Mean LOO RMS Cp error")
ax.set_title(f"Sensor placement algorithm comparison ({n_sensors} sensors)")
fig.tight_layout()
fig.savefig("placement_algorithm_comparison.png", dpi=200)
print("Saved placement_algorithm_comparison.png")

# ---------------------------------------------------------------------------
# PART 2: Noise robustness sweep
# Uses the greedy placement (your main method), trained once on the full
# ensemble, then tests reconstruction under increasing measurement noise
# at several sensor counts.
# ---------------------------------------------------------------------------
print("\n=== Noise robustness sweep ===")

U_r_full, X_mean_full, r_full = compute_pod(Cp_matrix, variance_target)
print(f"Full-ensemble POD: {r_full} modes retained")

noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]  # in Cp units
sensor_counts_for_noise = [5, 10, 20, 40]
n_trials = 20  # repeat each noise level to average out randomness

noise_results = []
rng = np.random.default_rng(123)

for n_sensors in sensor_counts_for_noise:
    selected = greedy_sensor_placement(U_r_full, n_sensors)
    for noise_sigma in noise_levels:
        trial_errors = []
        for trial in range(n_trials):
            # Test on a random held-out case with noise added to readings
            test_idx = rng.integers(0, n_cases)
            y_true = Cp_matrix[selected, test_idx]
            y_noisy = y_true + rng.normal(0, noise_sigma, size=len(selected))

            Cp_recon = reconstruct(U_r_full, X_mean_full, selected, y_noisy)
            err = np.sqrt(np.mean((Cp_recon - Cp_matrix[:, test_idx]) ** 2))
            trial_errors.append(err)

        noise_results.append({
            "n_sensors": n_sensors,
            "noise_sigma": noise_sigma,
            "mean_rms": np.mean(trial_errors),
            "std_rms": np.std(trial_errors),
        })
        print(f"  sensors={n_sensors}, noise sigma={noise_sigma}: "
              f"mean RMS={np.mean(trial_errors):.5f}")

noise_df = pd.DataFrame(noise_results)
noise_df.to_csv("noise_robustness_results.csv", index=False)
print("\nSaved noise_robustness_results.csv")

fig, ax = plt.subplots(figsize=(8, 6))
for n_sensors in sensor_counts_for_noise:
    subset = noise_df[noise_df["n_sensors"] == n_sensors]
    ax.plot(subset["noise_sigma"], subset["mean_rms"], "o-", label=f"{n_sensors} sensors")
ax.set_xlabel("Sensor measurement noise (Cp std dev)")
ax.set_ylabel("Reconstruction RMS Cp error")
ax.set_title("Noise robustness: error vs sensor count and noise level")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig("noise_robustness.png", dpi=200)
print("Saved noise_robustness.png")

print("\nAll done. Outputs: placement_algorithm_comparison.csv/.png, "
      "noise_robustness_results.csv/.png")