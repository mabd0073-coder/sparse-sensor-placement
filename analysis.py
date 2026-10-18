import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import plotly.graph_objects as go
# --- COMPATIBILITY PATCH FOR DEPRECATED NUMPY TYPES ---
legacy_aliases = [
    ('long', int), ('ulong', int), ('longlong', int), ('ulonglong', int),
    ('uintc', int), ('intc', int), ('int', int), ('float', float),
    ('complex', complex), ('object', object)
]
for alias, target in legacy_aliases:
    if not hasattr(np, alias):
        setattr(np, alias, target)
# ------------------------------------------------------


def get_colourmap(var):
    if var in ('CpT', 'yplus', 'ReasonForTermination'):
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
        for candidate in ["side_diffuser_case1", "side_diffuser_case1.txt", "side_diffuser.csv"]:
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError("Could not find suitable side diffuser input file.")
    
    cols = ["nodenumber", "x", "y", "z", "velocity_magnitude", "total_pressure",
            "dynamic_pressure", "pressure", "z2", "y2", "x2"]
    
    sep = r'\s+' if path.endswith('.txt') or not path.endswith('.csv') else ','
    df = pd.read_csv(path, sep=sep, header=0, names=cols, comment="#")
    df = df.sort_values("z").reset_index(drop=True)
    df["CpS"] = df["pressure"] / (0.5 * 1.225 * 19.44**2)
    return df


def plot_data(df, scalar_data, title, cmap_var="CpS", output_path="plot.png", 
              elev=20, azim=-60, close=True, sensors=None):
    cmap, norm = make_matplotlib_colormap(cmap_var)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    sc = ax.scatter(df["x"], df["y"], df["z"],
                    c=scalar_data, cmap=cmap, norm=norm, s=16, alpha=0.6)

    if sensors is not None:
        ax.scatter(sensors["x"], sensors["y"], sensors["z"],
                   c="red", s=100, marker="*", zorder=5, label=f"Sensors (n={len(sensors)})")
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


def greedy_sensor_placement(U_r, n_sensors, eps=1e-10, coords=None):
    n_points, r = U_r.shape
    selected = []
    remaining = list(range(n_points))
    
    DtD_inv = (1/eps) * np.eye(r)
    
    for k in range(n_sensors): 
        best_idx = None
        best_gain = -np.inf
        
        for i in remaining:
            w = U_r[i, :]
            gain = 1.0 + w @ DtD_inv @ w
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Rank-1 update (using .item() to fix deprecation warning)
        w = U_r[best_idx, :].reshape(-1, 1)
        Dw = DtD_inv @ w
        DtD_inv -= (Dw @ Dw.T) / (1.0 + (w.T @ Dw).item())
        
        coord_str = f"coords {coords[best_idx]}" if coords is not None else ""
        print(f"Sensor {k+1}: local candidate index {best_idx}, {coord_str}, gain {best_gain:.4f}")
    
    return selected


def reconstruct(U_r, X_mean, selected_indices, y_measured):
    """
    selected_indices: global indices of selected sensors (p,)
    y_measured: Cp values at sensor locations (p,)
    Returns full field reconstruction across ALL mesh nodes
    """
    D = U_r[selected_indices, :]   # (p, r)
    y = y_measured - X_mean[selected_indices, 0]
    
    z, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
    
    Cp_reconstructed = U_r @ z + X_mean[:, 0]
    return Cp_reconstructed

def plot_interactive_plotly(df, scalar_data, title="Interactive CpS", 
                            sensors=None, output_html="interactive_diffuser.html", max_points=50000):
    """
    Exports a 60fps GPU-accelerated 3D plot to an HTML file.
    Open the generated HTML in Chrome, Edge, or VS Code to rotate, zoom, and inspect points.
    """
    # Subsample if dataset is large to maintain smooth rotation
    if len(df) > max_points:
        sample_indices = np.random.choice(len(df), size=max_points, replace=False)
        df_plot = df.iloc[sample_indices]
        scalar_plot = scalar_data[sample_indices]
    else:
        df_plot = df
        scalar_plot = scalar_data

    fig = go.Figure()

    # 1. Surface Pressure Point Cloud
    fig.add_trace(go.Scatter3d(
        x=df_plot['x'],
        y=df_plot['y'],
        z=df_plot['z'],
        mode='markers',
        marker=dict(
            size=2.5,
            color=scalar_plot,
            colorscale='Jet',
            colorbar=dict(title="CpS"),
            opacity=0.8
        ),
        text=[f"X: {x:.3f}<br>Y: {y:.3f}<br>Z: {z:.3f}<br>CpS: {cp:.4f}" 
              for x, y, z, cp in zip(df_plot['x'], df_plot['y'], df_plot['z'], scalar_plot)],
        hoverinfo='text',
        name='Side Diffuser Surface'
    ))

    # 2. Overlay Sensor Locations (if provided)
    if sensors is not None:
        fig.add_trace(go.Scatter3d(
            x=sensors['x'],
            y=sensors['y'],
            z=sensors['z'],
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            text=[f"Sensor {i+1}" for i in range(len(sensors))],
            textposition="top center",
            hoverinfo='text+x+y+z',
            name='Optimal Sensors'
        ))

    # Equal 1:1 aspect ratio matching physical car dimensions
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(r=10, l=10, b=10, t=40)
    )

    fig.write_html(output_html)
    print(f"Saved interactive 3D model to {output_html}")
# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
df = load_data()
coords = df[['x', 'y', 'z']].values
CpS = df["CpS"].values.reshape(-1, 1)

# 1. LOAD EXCLUSION MASK & IDENTIFY VALID CANDIDATES
if os.path.exists("exclusion_mask.npz"):
    mask_data = np.load("exclusion_mask.npz")
    excluded_mask = mask_data["excluded"]
elif os.path.exists("all_points.csv") or os.path.exists("all_points_with_exclusion.csv"):
    csv_file = "all_points_with_exclusion.csv" if os.path.exists("all_points_with_exclusion.csv") else "all_points.csv"
    df_ex = pd.read_csv(csv_file)
    excluded_col = "excluded" if "excluded" in df_ex.columns else "near_reference"
    excluded_mask = df_ex[excluded_col].values.astype(bool)
else:
    print("Warning: No exclusion mask found. All points treated as valid candidates.")
    excluded_mask = np.zeros(len(df), dtype=bool)

candidate_indices = np.where(~excluded_mask)[0]
print(f"Excluded points: {excluded_mask.sum()} / {len(df)}")
print(f"Valid candidate points for placement: {len(candidate_indices)} / {len(df)}")

# 2. ENSEMBLE SETUP & POD (SVD)
n_snapshots = 200
noise_level = 0.1e-6
Cp_ensemble = np.tile(CpS, (1, n_snapshots))
Cp_ensemble += np.random.normal(0, noise_level, Cp_ensemble.shape)

X = Cp_ensemble - np.mean(Cp_ensemble, axis=0)
X_mean = X.mean(axis=1, keepdims=True)
X_tilde = X - X_mean

U, sigma, Vt = np.linalg.svd(X_tilde, full_matrices=False)

variance_lim = 0.9
cumulative = np.cumsum(sigma**2) / np.sum(sigma**2)
r = np.searchsorted(cumulative, variance_lim) + 1
print(f"Modes needed for {variance_lim*100}% variance: {r}")

U_r = U[:, :r]  # Full spatial basis: (n_all_points, r)

# 3. GREEDY PLACEMENT RESTRICTED TO CANDIDATES ONLY
U_r_candidates = U_r[candidate_indices, :]
coords_candidates = coords[candidate_indices]

n_sensors = 20  # Tune as needed
selected_local = greedy_sensor_placement(U_r_candidates, n_sensors, coords=coords_candidates)

# 4. MAP LOCAL CANDIDATE INDICES BACK TO GLOBAL MESH INDICES
selected_global = candidate_indices[selected_local]
sensors = df.iloc[selected_global]

print("\nSelected global sensor indices:", selected_global.tolist())
print("Selected sensor coordinates:")
print(sensors[['x', 'y', 'z']])

# 5. RECONSTRUCTION & ERROR EVALUATION
y_test = CpS[selected_global, 0]
Cp_recon = reconstruct(U_r, X_mean, selected_global, y_test)

error_rms = np.sqrt(np.mean((Cp_recon - CpS[:, 0])**2))
print(f"\nRMS reconstruction error: {error_rms:.6f}")

# 6. PLOTTING
sample_size = min(5000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)
error_abs = np.abs(Cp_recon - df["CpS"].values)

plot_data(df, df["CpS"].values, "Original CpS", output_path="original.png", cmap_var="CpS", elev=-90, azim=0)
plot_data(df, Cp_recon, "Reconstructed CpS", output_path="reconstructed.png", cmap_var="CpS", sensors=sensors, elev=-90, azim=0)
plot_data(df, error_abs, "Absolute Error", output_path="error.png", cmap_var="CpT", elev=-90, azim=0)
# Generate interactive plot with surface pressure and optimal sensors
plot_interactive_plotly(
    df, 
    Cp_recon, 
    title="Reconstructed CpS with Optimal Sensors", 
    sensors=sensors, 
    output_html="reconstructed_3d.html"
)
