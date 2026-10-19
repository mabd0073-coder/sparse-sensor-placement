import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def get_colourmap(var):
    if var == 'CpT' or var == 'yplus' or var == 'ReasonForTermination':
        output = [-1,0,0,0,-0.7,0.67,0.03,0.4,-0.65,0.73,0.035,0.45,-0.6,0.75,0.04,0.49,-0.55,0.74,0.05,0.52,-0.5,0.7,0.07,0.55,-0.25,0.29,0.3,0.78,-0.19999999999999996,0.2723,0.3697,0.8262,-0.15000000000000002,0.2765,0.4814,0.9506,-0.099999999999999978,0.2586,0.5796,0.9988,-0.050000000000000044,0.1933,0.6881,0.9619,0,0.127,0.7804,0.8759,0.050000000000000044,0.0929,0.8588,0.7724,0.10000000000000009,0.1353,0.922,0.6656,0.14999999999999991,0.248,0.9642,0.543,0.19999999999999996,0.4123,0.9927,0.3983,0.25,0.5603,0.9987,0.2862,0.30000000000000004,0.6746,0.9825,0.2196,0.35000000000000009,0.7856,0.9358,0.2034,0.39999999999999991,0.8753,0.8727,0.2156,0.44999999999999996,0.9449,0.7963,0.228,0.5,0.9878,0.7033,0.2136,0.55,0.9964,0.5985,0.1738,0.60000000000000009,0.9811,0.481,0.1233,0.64999999999999991,0.9408,0.3557,0.0703,0.7,0.8869,0.2615,0.0375,0.75,0.8161,0.1846,0.0181,0.8,0.7169,0.1127,0.0063,0.85000000000000009,0.7,0.15,0.05,0.89999999999999991,0.72,0.4,0.4,0.95,0.75,0.75,0.75,1,0.75,0.75,0.75]
    elif var == 'CpS':
        output = [-4,0,0,0,-3.25,0.67,0.03,0.4,-3.125,0.73,0.035,0.45,-3,0.75,0.04,0.49,-2.875,0.74,0.05,0.52,-2.75,0.7,0.07,0.55,-1.5,0.29,0.3,0.78,-1.25,0.2723,0.3697,0.8262,-1,0.2765,0.4814,0.9506,-0.75,0.2586,0.5796,0.9988,-0.5,0.1933,0.6881,0.9619,-0.42859999999999987,0.127,0.7804,0.8759,-0.3571,0.0929,0.8588,0.7724,-0.2856999999999994,0.1353,0.922,0.6656,-0.21419999999999995,0.248,0.9642,0.543,-0.14290000000000003,0.4123,0.9927,0.3983,-0.07140000000000013,0.5603,0.875,0.55,0,0.75,0.75,0.75,0.071399999999999686,0.7856,0.75,0.4,0.14290000000000003,0.8753,0.75,0.3,0.21419999999999995,0.9449,0.75,0.228,0.28570000000000029,0.9878,0.7033,0.2136,0.3571,0.9964,0.5985,0.1738,0.42860000000000031,0.9811,0.481,0.1233,0.5,0.9408,0.3557,0.0703,0.57139999999999969,0.8869,0.2615,0.0375,0.6429,0.8161,0.1846,0.0181,0.71429999999999971,0.7169,0.1127,0.0063,1,0,0,0]
    elif var in ('nV','heliQ','ndVx','ndVy','ndVz','JA'):
        output = [-1,0.75,0.04,0.49,-0.975,0.74,0.05,0.52,-0.95,0.7,0.07,0.55,-0.7,0.29,0.3,0.78,-0.65,0.2723,0.3697,0.8262,-0.6,0.2765,0.4814,0.9506,-0.55,0.2586,0.5796,0.9988,-0.5,0.1933,0.6881,0.9619,-0.42859999999999987,0.127,0.7804,0.8759,-0.3571,0.0929,0.8588,0.7724,-0.2856999999999994,0.1353,0.922,0.6656,-0.21419999999999995,0.248,0.9642,0.543,-0.14290000000000003,0.4123,0.9927,0.3983,-0.07140000000000013,0.5603,0.875,0.55,0,0.75,0.75,0.75,0.071399999999999686,0.7856,0.75,0.4,0.14290000000000003,0.8753,0.75,0.3,0.21419999999999995,0.9449,0.75,0.228,0.28570000000000029,0.9878,0.7033,0.2136,0.3571,0.9964,0.5985,0.1738,0.42860000000000031,0.9811,0.481,0.1233,0.5,0.9408,0.3557,0.0703,0.57139999999999969,0.8869,0.2615,0.0375,0.6429,0.8161,0.1846,0.0181,0.71429999999999971,0.7169,0.1127,0.0063,1,0,0,0]
    elif var == 'Cf':
        output = [0,0,1,1,0.002,0,0,1,0.004,0,0,0.5,0.006,1,0,0,0.008,1,1,0,0.01,1,1,1]
    elif var == 'Cfx':
        output = [-0.01,1,0.25,0.25,-9.9999999999995925E-6,1,0.25,0.25,0,0,1,1,0.01,0,1,1]
    else:
        raise ValueError(f"Unknown colourmap variable: {var}")
    return output

    
def make_matplotlib_colormap(var):
    output = get_colourmap(var)
    if len(output) % 4 != 0:
        raise ValueError("Colourmap data must be a multiple of 4")
    stops = [(output[i], output[i+1], output[i+2], output[i+3]) for i in range(0, len(output), 4)]
    positions, colors = zip(*[(pos, (r, g, b)) for pos, r, g, b in stops])
    vmin, vmax = positions[0], positions[-1]
    normalized = [(pos - vmin) / (vmax - vmin) for pos in positions]
    return LinearSegmentedColormap.from_list(var, list(zip(normalized, colors))), Normalize(vmin=vmin, vmax=vmax)


def load_data(path=None):
    if path is None:
        for candidate in [f"side_diffuser_case1", f"side_diffuser_case1.txt"]:
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(f"Could not find side_diffuser_case1 or side_diffuser_case1.txt")
    cols = [
        "nodenumber",
        "x",
        "y",
        "z",
        "velocity_magnitude",
        "total_pressure",
        "dynamic_pressure",
        "pressure",
        "z2",
        "y2",
        "x2",
    ]
    
    df = pd.read_csv(path, sep=r'\s+', header=0, names=cols, comment="#")
    df = df.sort_values("z")
    df["CpS"] = df["pressure"] / (0.5*1.225*19.44**2)  # Assuming air density and freestream velocity for normalization
    return df

def plot_data(df, scalar_data, title, cmap_var="CpS", 
              output_path="plot.png", elev=20, azim=-60, close=True ,sensors=None):
    cmap, norm = make_matplotlib_colormap(cmap_var)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    sc = ax.scatter(df["x"], df["y"], df["z"],
                    c=scalar_data, cmap=cmap, norm=norm, s=16, alpha=0.6)

    if sensors is not None:
        ax.scatter(sensors["x"],
                   sensors["y"],
                   sensors["z"],
                   c="red", s=100, marker="*",
                   zorder=5, label=f"Sensors (n={len(sensors)})")
        ax.legend()

    ax.view_init(elev=elev, azim=azim)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")
    if close:
        plt.close(fig)
    else:
        plt.show()
    
def greedy_sensor_placement(U_r, n_sensors, eps=1e-10,coords=[0,0,0]):
    n_points, r = U_r.shape
    selected = []
    remaining = list(range(n_points)) # CHANGE THIS TO THE LIST OF POINTS THAT SENSORS CAN BE PLACE ON
    
    # Initialize D^T D as small identity (eq. to Algorithm 2 in paper)
    DtD_inv = (1/eps) * np.eye(r)
    
    for k in range(n_sensors): 
        best_idx = None
        best_gain = -np.inf
        
        for i in remaining:
            w = U_r[i, :]          # (r,) — this point's mode values
            # Woodbury identity: gain = det(I + w (DtD)^-1 w^T)
            # For scalar w: gain = 1 + w^T DtD_inv w
            gain = 1.0 + w @ DtD_inv @ w
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Rank-1 update of (D^T D)^-1 via Sherman-Morrison
        w = U_r[best_idx, :].reshape(-1, 1)
        Dw = DtD_inv @ w
        DtD_inv -= (Dw @ Dw.T) / (1.0 + float(w.T @ Dw))
        
        print(f"Sensor {k+1}: point index {best_idx}, "
              f"coords {coords[best_idx]}, gain {best_gain:.4f}")
    
    return selected

# ... [Keep your get_colourmap, make_matplotlib_colormap, plot_data, and greedy_sensor_placement functions here] ...

def load_multiple_cases(file_pattern="case_data/side_diffuser_case*.txt", default_v_inf=19.44, rho=1.225):
    """
    Loads multiple CFD cases into a single snapshot matrix.
    Assumes all cases share the same mesh topology (node numbers and local coordinates).
    """
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching {file_pattern}")
    
    print(f"Found {len(files)} cases. Building snapshot matrix...")
    
    Cp_list = []
    coords = None
    template_df = None
    
    for f in files:
        cols = ["nodenumber", "x", "y", "z", "velocity_magnitude", 
                "total_pressure", "dynamic_pressure", "pressure", "z2", "y2", "x2"]
        df = pd.read_csv(f, sep=r'\s+', header=0, names=cols, comment="#")
        
        # CRITICAL: Sort by nodenumber to guarantee spatial alignment across all cases
        df = df.sort_values("nodenumber")
        
        # Extract base coordinates from the first file
        if coords is None:
            coords = df[['x', 'y', 'z']].values
            template_df = df.copy() 
        
        # If your cases use different freestream velocities (e.g., speed sweeps), 
        # extract v_inf from the file metadata. Otherwise, use default.
        v_inf = default_v_inf 
        
        # Calculate non-dimensional Cp
        Cp = df["pressure"].values / (0.5 * rho * v_inf**2)
        Cp_list.append(Cp)
        
    # Snapshot matrix X: Rows = Spatial points (n), Columns = Cases (m)
    # Shape: (n_points, n_cases)
    X = np.column_stack(Cp_list)
    
    return X, coords, template_df

# ==========================================
# 1. Load Data & Build Snapshot Matrix
# ==========================================
# Point this to a folder containing all your sweep text files
X, coords, df = load_multiple_cases("side_diffuser_case*.txt")
n_points, n_cases = X.shape
print(f"Snapshot matrix shape: {n_points} points x {n_cases} cases")

# ==========================================
# 2. Proper Orthogonal Decomposition (POD)
# ==========================================
X_mean = X.mean(axis=1, keepdims=True)
X_tilde = X - X_mean

# Economy SVD (returns min(n_points, n_cases) modes)
U, sigma, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# Dynamically select modes based on variance
variance_lim = 0.99  # Multi-case datasets require higher limits (95-99%) to capture asymmetries
cumulative = np.cumsum(sigma**2) / np.sum(sigma**2)
r = int(np.searchsorted(cumulative, variance_lim) + 1)
r = min(r, U.shape[1]) # Cap at max available modes
print(f"Modes needed for {variance_lim*100}% variance: {r} (out of {n_cases})")

U_r = U[:, :r]  # Shape: (n_points, r)

# ==========================================
# 3. Greedy Sensor Placement
# ==========================================
n_sensors = 20  # Rule of thumb: n_sensors should be slightly >= r
selected_indices = greedy_sensor_placement(U_r, n_sensors, coords=coords)
sensors = df.iloc[selected_indices]

# ==========================================
# 4. Reconstruction & Evaluation
# ==========================================
def reconstruct(U_r, X_mean, selected_indices, y_measured):
    """Reconstructs the full field from sparse sensor measurements."""
    D = U_r[selected_indices, :]   # (p, r)
    y = y_measured - X_mean[selected_indices, 0]
    z, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
    return U_r @ z + X_mean[:, 0]

# Evaluate reconstruction across ALL cases
rms_errors = []
for case_idx in range(n_cases):
    # Extract ground truth for this specific case
    Cp_true = X[:, case_idx]
    
    # Simulate what the sensors would read for this case
    y_test = Cp_true[selected_indices] 
    
    # Reconstruct full field
    Cp_recon = reconstruct(U_r, X_mean, selected_indices, y_test)
    
    # Calculate error
    case_error = np.sqrt(np.mean((Cp_recon - Cp_true)**2))
    rms_errors.append(case_error)

print(f"\nAverage RMS error across all {n_cases} cases: {np.mean(rms_errors):.6f}")
print(f"Worst-case RMS error: {np.max(rms_errors):.6f} (Case {np.argmax(rms_errors)})")

# Plot worst-case reconstruction
worst_case_idx = np.argmax(rms_errors)
Cp_true_worst = X[:, worst_case_idx]
Cp_recon_worst = reconstruct(U_r, X_mean, selected_indices, Cp_true_worst[selected_indices])
error_worst = np.abs(Cp_recon_worst - Cp_true_worst)

plot_data(df, Cp_true_worst, f"Original CpS (Worst Case {worst_case_idx})", output_path="original_worst.png", cmap_var="CpS", elev=-90, azim=0)
plot_data(df, Cp_recon_worst, f"Reconstructed CpS (Worst Case {worst_case_idx})", output_path="recon_worst.png", cmap_var="CpS", sensors=sensors, elev=-90, azim=0)
plot_data(df, error_worst, "Absolute Error (Worst Case)", output_path="error_worst.png", cmap_var="CpT", elev=-90, azim=0)