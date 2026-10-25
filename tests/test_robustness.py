import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import robustness as rb  # noqa: E402


def orthonormal_basis(rng, n, r):
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return U


def test_predicted_noise_rms_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n, r, k, sigma = 500, 5, 12, 0.03
    U = orthonormal_basis(rng, n, r)
    mean = rng.standard_normal((n, 1))
    fields = mean + U @ rng.standard_normal((r, 3))
    s = list(rng.choice(n, k, replace=False))

    predicted = rb.predicted_noise_rms(U, s, sigma)
    measured = rb.monte_carlo_noise_rms(U, mean, s, fields, sigma, draws=2000, rng=rng)

    assert abs(measured - predicted) / predicted < 0.10


def test_pod_gram_matches_pod_svd_on_well_conditioned_matrix():
    rng = np.random.default_rng(1)
    Q1, _ = np.linalg.qr(rng.standard_normal((200, 10)))
    Q2, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    Xc = Q1 @ np.diag(np.linspace(10.0, 1.0, 10)) @ Q2  # condition number 10

    _, s_svd = rb.pod_svd(Xc)
    U_gram, s_gram = rb.pod_gram(Xc)

    np.testing.assert_allclose(s_gram, s_svd, rtol=1e-8)
    np.testing.assert_allclose(U_gram.T @ U_gram, np.eye(10), atol=1e-8)


def test_residual_is_zero_for_field_in_span_of_U():
    rng = np.random.default_rng(2)
    n, r = 400, 6
    U = orthonormal_basis(rng, n, r)
    mean = rng.standard_normal((n, 1))
    x = mean[:, 0] + U @ rng.standard_normal(r)
    s = list(rng.choice(n, 20, replace=False))

    assert rb.residual(U, mean, s, x) < 1e-12
