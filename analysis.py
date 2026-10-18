"""
SVD-Based Sparse Sensor Placement and Flow Field Reconstruction Framework.

This module performs Proper Orthogonal Decomposition (POD / SVD) on an ensemble
of 3D aerodynamic pressure fields, applies a greedy sensor selection algorithm
with spatial separation constraints, and reconstructs high-fidelity pressure fields
from sparse surface measurements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import cKDTree


def greedy_sensor_placement(
    U_r: np.ndarray, 
    n_sensors: int, 
    coords: np.ndarray | None = None, 
    min_distance: float = 0.03
) -> np.ndarray:
    """
    Select optimal sensor locations using a greedy matrix volume maximization algorithm.

    Iteratively selects sensor locations that maximize projection residual norm
    from the truncated spatial modes $U_r$, while enforcing a physical minimum distance
    between selected sensors.

    Parameters
    ----------
    U_r : np.ndarray
        Truncated left singular vector matrix (spatial modes) of shape (N_points, r).
    n_sensors : int
        Number of sensors to select.
    coords : np.ndarray, optional
        3D spatial coordinates of candidate points of shape (N_points, 3) in meters.
    min_distance : float, optional
        Minimum physical separation required between selected sensors in meters (default 0.03 m).

    Returns
    -------
    np.ndarray
        1D array containing indices of the selected sensor positions.
    """
    n_points, r = U_r.shape
    selected_indices = []

    # First sensor selection: location with maximum mode amplitude norm
    norms = np.linalg.norm(U_r, axis=1)
    first_idx = int(np.argmax(norms))
    selected_indices.append(first_idx)

    for _ in range(1, n_sensors):
        U_sub = U_r[selected_indices, :]

        if U_sub.shape[0] < r:
            P = U_sub.T @ np.linalg.pinv(U_sub @ U_sub.T) @ U_sub
            residuals = np.linalg.norm(U_r - U_r @ P, axis=1)
        else:
            residuals = np.linalg.norm(U_r, axis=1)

        # Apply minimum spatial separation constraint
        if coords is not None and len(selected_indices) > 0:
            tree = cKDTree(coords[selected_indices])
            dists, _ = tree.query(coords)
            residuals[dists < min_distance] = -1.0

        next_idx = int(np.argmax(residuals))
        selected_indices.append(next_idx)

    return np.array(selected_indices, dtype=int)


def reconstruct(
    U_r: np.ndarray, 
    Cp_mean: np.ndarray, 
    sensor_indices: np.ndarray, 
    y_sensors: np.ndarray
) -> np.ndarray:
    """
    Reconstruct full 3D surface pressure field from sparse sensor measurements.

    Solves the linear system $y' = U_{s} a$ for modal coefficients $a$ using least squares:
    $$a = U_s^\dagger (y_{\text{sensors}} - \bar{y}_{\text{sensors}})$$
    and computes full field estimate: $\hat{y} = \bar{y} + U_r a$.

    Parameters
    ----------
    U_r : np.ndarray
        Truncated spatial modes matrix of shape (N_points, r).
    Cp_mean : np.ndarray
        Ensemble mean field array of shape (N_points, 1) or (N_points,).
    sensor_indices : np.ndarray
        Global indices corresponding to selected sensor locations.
    y_sensors : np.ndarray
        Measured pressure coefficient values at sensor locations.

    Returns
    -------
    np.ndarray
        Reconstructed full surface pressure field array of shape (N_points,).
    """
    U_s = U_r[sensor_indices, :]
    y_prime = y_sensors - Cp_mean[sensor_indices].flatten()

    # Least-squares modal expansion coefficient solving
    a, _, _, _ = np.linalg.lstsq(U_s, y_prime, rcond=None)

    # Full field projection
    Cp_recon = Cp_mean.flatten() + (U_r @ a)
    return Cp_recon


def plot_data(
    df: pd.DataFrame, 
    values: np.ndarray, 
    title: str, 
    output_path: str, 
    cmap_var: str = "coolwarm", 
    sensors: pd.DataFrame | None = None, 
    vmin: float = -3.0, 
    vmax: float = 1.0
) -> None:
    """
    Generate and save a static 3D scatter plot of surface field data using Matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 3D mesh spatial coordinates 'x', 'y', 'z'.
    values : np.ndarray
        Scalar array to visualize with color mapping.
    title : str
        Plot title text.
    output_path : str
        Target file path for saving the rendered image file (.png, .jpg).
    cmap_var : str, optional
        Matplotlib color map identifier (default is "coolwarm").
    sensors : pd.DataFrame, optional
        DataFrame containing selected sensor coordinates for overlay marking.
    vmin : float, optional
        Minimum color scale value (default -3.0).
    vmax : float, optional
        Maximum color scale value (default 1.0).
    """
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        df['y'], df['x'], df['z'],
        c=values,
        cmap=cmap_var,
        s=4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8
    )

    if sensors is not None:
        ax.scatter(
            sensors['y'], sensors['x'], sensors['z'],
            c='red', marker='*', s=120, label='Sensors', edgecolors='black'
        )
        ax.legend(loc='upper right')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Y [m]')
    ax.set_ylabel('X [m]')
    ax.set_zlabel('Z [m]')

    ax.view_init(elev=-90, azim=0)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_interactive_plotly(
    df: pd.DataFrame, 
    values: np.ndarray, 
    title: str, 
    sensors: pd.DataFrame | None = None, 
    output_html: str = "reconstructed_3d.html"
) -> None:
    """
    Export an interactive 3D WebGL scatter visualization using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 3D mesh spatial coordinates 'x', 'y', 'z'.
    values : np.ndarray
        Scalar array mapped to marker color values.
    title : str
        Plotly figure title.
    sensors : pd.DataFrame, optional
        DataFrame containing sensor coordinates to overlay on the geometry.
    output_html : str, optional
        Output HTML filename (default is "reconstructed_3d.html").
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df['y'], y=df['x'], z=df['z'],
        mode='markers',
        marker=dict(
            size=3,
            color=values,
            colorscale='Jet',
            cmin=-3.0,
            cmax=1.0,
            colorbar=dict(title="CpS"),
            opacity=0.8
        ),
        name='Surface Mesh'
    ))

    if sensors is not None:
        fig.add_trace(go.Scatter3d(
            x=sensors['y'], y=sensors['x'], z=sensors['z'],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            name='Sensors'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Y [m]',
            yaxis_title='X [m]',
            zaxis_title='Z [m]',
            aspectmode='data'
        )
    )
    fig.write_html(output_html)
    print(f"Saved interactive plot to {output_html}")


if __name__ == "__main__":
    ENSEMBLE_PATH = "side_diffuser_ensemble_combined.npz"
    MASK_PATH = "exclusion_mask.npz"

    if not os.path.exists(ENSEMBLE_PATH):
        raise FileNotFoundError(f"Cannot find {ENSEMBLE_PATH}. Run interpolation pipeline first.")

    # 1. Load data archive
    data = np.load(ENSEMBLE_PATH)
    Cp_ensemble = data["Cp_matrix"]
    x, y, z = data["x"], data["y"], data["z"]
    coords = np.column_stack([x, y, z])
    Cp_mean = Cp_ensemble.mean(axis=1, keepdims=True)

    # 2. Deduplicate physical mesh nodes
    _, unique_indices = np.unique(np.round(coords, decimals=4), axis=0, return_index=True)
    unique_indices.sort()

    coords_unique = coords[unique_indices]
    Cp_ensemble_unique = Cp_ensemble[unique_indices, :]
    Cp_mean_unique = Cp_mean[unique_indices]

    df = pd.DataFrame({
        "x": coords_unique[:, 0],
        "y": coords_unique[:, 1],
        "z": coords_unique[:, 2],
        "CpS": Cp_mean_unique.flatten()
    })

    # 3. Apply exclusion mask
    if os.path.exists(MASK_PATH):
        mask_data = np.load(MASK_PATH)
        raw_mask = mask_data["excluded"]

        if len(raw_mask) != len(df):
            mask_coords = np.column_stack([mask_data["x"], mask_data["y"], mask_data["z"]]) if "x" in mask_data else coords[:len(raw_mask)]
            if np.max(np.abs(mask_coords)) > 10.0:
                mask_coords /= 1000.0

            tree = cKDTree(mask_coords)
            _, nearest_idx = tree.query(coords_unique)
            excluded_mask = raw_mask[nearest_idx].astype(bool)
        else:
            excluded_mask = raw_mask[unique_indices].astype(bool)
    else:
        excluded_mask = np.zeros(len(df), dtype=bool)

    candidate_indices = np.where(~excluded_mask)[0]

    print(f"Total Unique Mesh Points: {len(df)}")
    print(f"Candidate Points for Sensor Placement: {len(candidate_indices)}")

    # 4. Perform SVD
    X_tilde = Cp_ensemble_unique - Cp_mean_unique
    U, sigma, Vt = np.linalg.svd(X_tilde, full_matrices=False)

    variance_target = 0.95
    cumulative_var = np.cumsum(sigma**2) / np.sum(sigma**2)
    r = max(2, int(np.searchsorted(cumulative_var, variance_target) + 1))

    print(f"SVD Modes selected for {variance_target*100:.0f}% variance: {r}")
    U_r = U[:, :r]

    # 5. Greedy sensor selection
    n_sensors = 20
    selected_local = greedy_sensor_placement(
        U_r[candidate_indices, :],
        n_sensors=n_sensors,
        coords=coords_unique[candidate_indices],
        min_distance=0.03
    )

    selected_global = candidate_indices[selected_local]
    sensors = df.iloc[selected_global]

    print("\nSelected Sensor Coordinates (Meters):")
    print(sensors[['x', 'y', 'z']])

    # 6. Evaluate reconstruction on Case 0
    test_case_idx = 0
    y_test_case = Cp_ensemble_unique[selected_global, test_case_idx]
    Cp_true_case = Cp_ensemble_unique[:, test_case_idx]

    Cp_recon = reconstruct(U_r, Cp_mean_unique, selected_global, y_test_case)

    rms_error = float(np.sqrt(np.mean((Cp_recon - Cp_true_case)**2)))
    print(f"\nRMS Reconstruction Error (Case {test_case_idx}): {rms_error:.6f}")

    # 7. Render output plots
    plot_data(df, Cp_true_case, f"True CpS (Case {test_case_idx})", "original.png", cmap_var="jet", vmin=-3.0, vmax=1.0)
    plot_data(df, Cp_recon, f"Reconstructed CpS (Case {test_case_idx})", "reconstructed.png", cmap_var="jet", sensors=sensors, vmin=-3.0, vmax=1.0)
    plot_data(df, np.abs(Cp_recon - Cp_true_case), "Absolute Reconstruction Error", "error.png", cmap_var="hot", vmin=0.0, vmax=0.5)

    plot_interactive_plotly(
        df,
        Cp_recon,
        title=f"Reconstructed CpS (Case {test_case_idx}) with 20 Optimal Sensors",
        sensors=sensors,
        output_html="reconstructed_3d.html"
    )