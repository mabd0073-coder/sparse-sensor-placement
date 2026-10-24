"""
POD basis construction, greedy D-optimal sensor placement and sparse reconstruction.

The workflow is:

1. :func:`compute_pod` builds a reduced basis ``U_r`` from an ensemble of
   pressure fields (one column per CFD case).
2. :func:`greedy_sensor_placement` picks the rows of ``U_r`` (mesh points)
   that make the sensor matrix best conditioned.
3. :func:`reconstruct` recovers the full field from readings at those points.
"""

import numpy as np


class CfdDataError(ValueError):
    """Raised when CFD input data contains NaN or infinite values."""


def _check_finite(name, array):
    if not np.all(np.isfinite(array)):
        n_bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise CfdDataError(f"{name} contains {n_bad} NaN or infinite value(s).")


def compute_pod(snapshots, variance_target=0.90, min_modes=2):
    """Compute a truncated POD basis from an ensemble of fields.

    Parameters
    ----------
    snapshots : ndarray, shape (n_points, n_cases)
        Field values, one column per case.
    variance_target : float, optional
        Fraction of the fluctuation energy (sum of squared singular values)
        the retained modes must capture. Default 0.90.
    min_modes : int, optional
        Lower bound on the number of retained modes. Default 2.

    Returns
    -------
    U_r : ndarray, shape (n_points, r)
        Leading ``r`` left singular vectors of the mean-subtracted ensemble.
    mean : ndarray, shape (n_points, 1)
        Ensemble mean field.
    r : int
        Number of retained modes, ``max(min_modes, k)`` where ``k`` is the
        smallest count reaching ``variance_target``.

    Raises
    ------
    CfdDataError
        If ``snapshots`` contains NaN or infinite values.
    """
    snapshots = np.asarray(snapshots)
    _check_finite("snapshots", snapshots)

    mean = snapshots.mean(axis=1, keepdims=True)
    U, sigma, _ = np.linalg.svd(snapshots - mean, full_matrices=False)
    cumulative = np.cumsum(sigma**2) / np.sum(sigma**2)
    r = int(max(min_modes, np.searchsorted(cumulative, variance_target) + 1))
    return U[:, :r], mean, r


def greedy_sensor_placement(U_r, n_sensors, eps=1e-10, coords=None, verbose=False):
    """Select sensor locations by greedy D-optimal design.

    At each step the point whose basis row ``w`` maximises the determinant
    gain ``1 + w (D^T D + eps I)^-1 w^T`` is added, and the inverse is
    updated with the Sherman-Morrison formula.

    Parameters
    ----------
    U_r : ndarray, shape (n_points, r)
        Basis restricted to the candidate points.
    n_sensors : int
        Number of sensors to place. Must not exceed ``n_points``.
    eps : float, optional
        Regularisation of the initial information matrix. Default 1e-10.
    coords : ndarray, shape (n_points, 3), optional
        Candidate coordinates, only used in progress messages.
    verbose : bool, optional
        Print each selected sensor. Default False.

    Returns
    -------
    list of int
        Row indices into ``U_r`` in selection order; all distinct.

    Raises
    ------
    CfdDataError
        If ``U_r`` contains NaN or infinite values.
    ValueError
        If ``n_sensors`` exceeds the number of candidate points.
    """
    U_r = np.asarray(U_r)
    _check_finite("U_r", U_r)
    n_points, r = U_r.shape
    if n_sensors > n_points:
        raise ValueError(f"Cannot place {n_sensors} sensors on {n_points} candidate points.")

    selected = []
    available = np.ones(n_points, dtype=bool)

    DtD_inv = (1/eps) * np.eye(r)

    for k in range(n_sensors):
        gains = 1.0 + ((U_r @ DtD_inv) * U_r).sum(axis=1)
        gains[~available] = -np.inf
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]

        selected.append(best_idx)
        available[best_idx] = False

        w = U_r[best_idx, :].reshape(-1, 1)
        Dw = DtD_inv @ w
        DtD_inv -= (Dw @ Dw.T) / (1.0 + (w.T @ Dw).item())

        if verbose:
            coord_str = f"coords {coords[best_idx]}" if coords is not None else ""
            print(f"Sensor {k+1}: local candidate index {best_idx}, {coord_str}, gain {best_gain:.4f}")

    return selected


def reconstruct(U_r, X_mean, selected_indices, y_measured):
    """Reconstruct a full field from sparse sensor readings.

    Solves ``min_z || U_r[selected] z - (y - mean[selected]) ||`` by least
    squares and returns ``U_r z + mean``.

    Parameters
    ----------
    U_r : ndarray, shape (n_points, r)
        POD basis over the full mesh.
    X_mean : ndarray, shape (n_points, 1)
        Mean field the basis was built around.
    selected_indices : array_like of int
        Mesh indices of the sensors.
    y_measured : ndarray, shape (n_sensors,)
        Readings at ``selected_indices``.

    Returns
    -------
    ndarray, shape (n_points,)
        Reconstructed field.

    Raises
    ------
    CfdDataError
        If ``y_measured`` contains NaN or infinite values.
    """
    _check_finite("y_measured", y_measured)
    D = U_r[selected_indices, :]
    y = y_measured - X_mean[selected_indices, 0]
    z, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
    Cp_reconstructed = U_r @ z + X_mean[:, 0]
    return Cp_reconstructed
