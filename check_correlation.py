"""
check_correlation.py

Two diagnostics on the real 100-case ride-height ensemble:
1. Does the dominant POD mode correlate with combined ride height
   (FRH + RRH)? A high correlation supports "1 mode is a real physical
   finding" rather than a data artifact.
2. Leave-one-out cross-validation: train POD + sensor placement on 99
   cases, reconstruct the held-out case, and measure error. This is the
   real test of whether "1 mode / few sensors" generalizes, rather than
   just fitting the training data it was built from.

Run this from the same folder as side_diffuser_ensemble_dedup.npz and
case_params.csv (i.e. wherever run_real_pipeline.py already ran).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the deduplicated real ensemble
# ---------------------------------------------------------------------------
data = np.load("side_diffuser_ensemble_dedup.npz")
Cp_matrix = data["Cp_matrix"]                      # (N, 100)
coords = np.column_stack([data["x"], data["y"], data["z"]])
case_params = pd.read_csv("case_params.csv")

print(f"Ensemble shape: {Cp_matrix.shape}")

# ---------------------------------------------------------------------------
# Check 1: correlation between mean Cp per case and combined ride height
# ---------------------------------------------------------------------------
mean_cp_per_case = Cp_matrix.mean(axis=0)
case_params = case_params.copy()
case_params["mean_cp"] = mean_cp_per_case

print("\nCases sorted by mean Cp:")
print(case_params[["folder", "FRH", "RRH", "mean_cp"]].sort_values("mean_cp").to_string())

combined_rh = case_params["FRH"] + case_params["RRH"]
corr = np.corrcoef(combined_rh, mean_cp_per_case)[0, 1]
print(f"\nCorrelation between (FRH + RRH) and mean Cp across cases: {corr:.4f}")

# Also check correlation with the actual first POD mode coefficient (Vt row 0),
# which is more directly what "1 mode explains everything" is claiming.
X_mean = Cp_matrix.mean(axis=1, keepdims=True)
X_tilde = Cp_matrix - X_mean
U, sigma, Vt = np.linalg.svd(X_tilde, full_matrices=False)
mode1_coeff = Vt[0, :]  # how much of mode 1 each case needs

corr_mode1 = np.corrcoef(combined_rh, mode1_coeff)[0, 1]
print(f"Correlation between (FRH + RRH) and mode-1 coefficient: {corr_mode1:.4f}")
print("(Values close to +-1 support a genuine single dominant physical mode;")
print(" values well below that suggest the '1 mode' result may not generalize.)")

# ---------------------------------------------------------------------------
# Check 2: leave-one-out cross-validation
# ---------------------------------------------------------------------------
def greedy_sensor_placement(U_r, n_sensors, eps=1e-10, verbose=False):
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
        if verbose:
            print(f"  Sensor {k+1}: point {best_idx}, gain {best_gain:.4f}")

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


n_cases = Cp_matrix.shape[1]
n_sensors = 10  # fewer sensors than before, since we expect very low rank
variance_target = 0.9

print(f"\nRunning leave-one-out validation ({n_cases} folds, "
      f"{n_sensors} sensors, {variance_target*100:.0f}% variance target)...")
print("This may take a little while (full SVD run 100 times)...\n")

loo_results = []
for test_idx in range(n_cases):
    train_cols = [i for i in range(n_cases) if i != test_idx]
    Cp_train = Cp_matrix[:, train_cols]
    Cp_test = Cp_matrix[:, test_idx]

    U_r, X_mean_train, r = compute_pod(Cp_train, variance_target)
    selected = greedy_sensor_placement(U_r, n_sensors)

    y_test = Cp_test[selected]
    Cp_recon = reconstruct(U_r, X_mean_train, selected, y_test)
    err = np.sqrt(np.mean((Cp_recon - Cp_test) ** 2))

    folder = case_params.iloc[test_idx]["folder"]
    loo_results.append({"held_out_case": folder, "n_modes": r, "rms_error": err})
    print(f"[{test_idx+1:3d}/{n_cases}] held out {folder}: modes={r}, RMS error={err:.5f}")

loo_df = pd.DataFrame(loo_results)
loo_df.to_csv("loo_results.csv", index=False)

print("\n=== Leave-one-out summary ===")
print(loo_df["rms_error"].describe())
print(f"\nWorst 5 held-out cases:")
print(loo_df.sort_values("rms_error", ascending=False).head())
print("\nSaved loo_results.csv")