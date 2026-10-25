import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import feasible_layout as fl  # noqa: E402


def random_problem(seed, n=1500, r=6):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, r))
    coords = rng.uniform(0.0, 1.0, size=(n, 3))
    allowed = rng.random(n) < 0.6
    return U, coords, allowed


def pairwise_min(coords, sel):
    p = coords[sel]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return d[np.triu_indices(len(sel), k=1)].min()


def test_greedy_respects_allowed_mask_and_min_dist():
    U, coords, allowed = random_problem(0)

    sel = fl.greedy(U, 24, coords, allowed, min_dist=0.1)

    assert len(sel) == 24
    assert len(set(sel)) == 24
    assert allowed[sel].all()
    assert pairwise_min(coords, sel) >= 0.1


def test_exchange_never_decreases_log_det_and_stays_feasible():
    for seed in range(5):
        U, coords, allowed = random_problem(seed)
        start = fl.greedy(U, 24, coords, allowed, min_dist=0.08)

        swapped = fl.exchange(U, start, coords, allowed, min_dist=0.08)

        assert fl.logdet(U, swapped) >= fl.logdet(U, start) - 1e-9
        assert allowed[swapped].all()
        assert len(set(swapped)) == len(swapped)
        assert pairwise_min(coords, swapped) >= 0.08
