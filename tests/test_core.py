import numpy as np
import pytest

from sensor_opt.core import CfdDataError, compute_pod, greedy_sensor_placement, reconstruct


def synthetic_ensemble(rng, n_points, n_cases, rank):
    """Fields base + Q @ A with orthonormal Q of the given rank; returns (snapshots, base, Q)."""
    Q, _ = np.linalg.qr(rng.standard_normal((n_points, rank)))
    scales = np.linspace(10.0, 1.0, rank)[:, None]
    A = scales * rng.standard_normal((rank, n_cases))
    base = rng.standard_normal((n_points, 1))
    return base + Q @ A, base, Q


def reference_greedy(U_r, n_sensors, eps=1e-10):
    """Original loop implementation of greedy_sensor_placement, kept as an oracle."""
    n_points, r = U_r.shape
    selected = []
    remaining = list(range(n_points))
    DtD_inv = (1/eps) * np.eye(r)
    for _ in range(n_sensors):
        best_idx, best_gain = None, -np.inf
        for i in remaining:
            w = U_r[i, :]
            gain = 1.0 + w @ DtD_inv @ w
            if gain > best_gain:
                best_gain, best_idx = gain, i
        selected.append(best_idx)
        remaining.remove(best_idx)
        w = U_r[best_idx, :].reshape(-1, 1)
        Dw = DtD_inv @ w
        DtD_inv -= (Dw @ Dw.T) / (1.0 + (w.T @ Dw).item())
    return selected


@pytest.mark.parametrize("rank", [1, 3, 8])
def test_exact_recovery_of_low_rank_field(rank):
    rng = np.random.default_rng(rank)
    snapshots, base, Q = synthetic_ensemble(rng, n_points=400, n_cases=30, rank=rank)

    U_r, mean, r = compute_pod(snapshots, variance_target=1 - 1e-9)
    assert r == max(2, rank)

    sensors = greedy_sensor_placement(U_r, n_sensors=r + 2)
    truth = (base + Q @ rng.standard_normal((rank, 1)))[:, 0]  # unseen case in the same subspace
    recon = reconstruct(U_r, mean, sensors, truth[sensors])

    assert np.max(np.abs(recon - truth)) < 1e-10


def test_greedy_returns_unique_indices():
    rng = np.random.default_rng(0)
    U_r = rng.standard_normal((500, 6))

    selected = greedy_sensor_placement(U_r, n_sensors=60)

    assert len(selected) == 60
    assert len(set(selected)) == 60
    assert all(0 <= i < 500 for i in selected)


def test_greedy_rejects_more_sensors_than_points():
    with pytest.raises(ValueError):
        greedy_sensor_placement(np.ones((5, 2)), n_sensors=6)


def test_vectorised_greedy_matches_original_loop():
    rng = np.random.default_rng(42)
    U_r = rng.standard_normal((2000, 5))

    assert greedy_sensor_placement(U_r, n_sensors=30) == reference_greedy(U_r, n_sensors=30)


def test_reconstruction_error_falls_with_more_sensors_under_noise():
    rng = np.random.default_rng(7)
    snapshots, base, Q = synthetic_ensemble(rng, n_points=1000, n_cases=40, rank=5)
    U_r, mean, r = compute_pod(snapshots, variance_target=1 - 1e-9)
    order = greedy_sensor_placement(U_r, n_sensors=40)

    sensor_counts = [5, 10, 20, 40]
    errors = np.zeros(len(sensor_counts))
    n_trials = 200
    for _ in range(n_trials):
        truth = (base + Q @ rng.standard_normal((5, 1)))[:, 0]
        noisy = truth + 0.05 * rng.standard_normal(truth.shape)
        for j, n in enumerate(sensor_counts):
            sensors = order[:n]
            recon = reconstruct(U_r, mean, sensors, noisy[sensors])
            errors[j] += np.sqrt(np.mean((recon - truth) ** 2)) / n_trials

    assert np.all(np.diff(errors) < 0), errors


@pytest.mark.parametrize("call", ["compute_pod", "greedy", "reconstruct"])
def test_nan_input_raises_cfd_data_error(call):
    rng = np.random.default_rng(1)
    snapshots, _, _ = synthetic_ensemble(rng, n_points=50, n_cases=10, rank=3)
    U_r, mean, _ = compute_pod(snapshots)

    with pytest.raises(CfdDataError):
        if call == "compute_pod":
            snapshots[3, 2] = np.nan
            compute_pod(snapshots)
        elif call == "greedy":
            U_r[7, 0] = np.nan
            greedy_sensor_placement(U_r, n_sensors=4)
        else:
            y = snapshots[:4, 0].copy()
            y[1] = np.nan
            reconstruct(U_r, mean, [0, 1, 2, 3], y)
