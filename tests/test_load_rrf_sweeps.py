import sys

import numpy as np

from cluster import load_rrf_sweeps

HEADER = "nodenumber x y z pressure dynamic_pressure total_pressure CpS velocity_magnitude\n"


def write_case(case_dir, coords, cp):
    (case_dir / "out").mkdir(parents=True)
    lines = [HEADER] + [
        f"{i} {x} {y} {z} 0 0 0 {c} 0\n" for i, ((x, y, z), c) in enumerate(zip(coords, cp))
    ]
    (case_dir / "out" / "side_diffuser.csv").write_text("".join(lines))


def test_builds_left_side_train_and_test_ensembles(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    left = np.column_stack([rng.uniform(0.3, 1.2, 60), rng.uniform(-0.7, -0.25, 60), rng.uniform(0, 0.2, 60)])
    both_sides = np.vstack([left, left * [1, -1, 1]])  # full car: left + right
    cp_full = rng.standard_normal(120)

    write_case(tmp_path / "50.39" / "1_FRH20.00_RRH35.00_R-0.84_S0.00_CR500.00_Y2.51_V19.44", both_sides, cp_full)
    (tmp_path / "50.39" / "2_FRH30.00_RRH30.00_R0.00_S0.00_CR500.00_Y0.00_V19.44").mkdir(parents=True)  # no CSV
    write_case(tmp_path / "50.41" / "1_FRH40.00_RRH25.00_R1.50_S10.00_CR8.00_Y-1.00_V12.00",
               left * [1, -1, 1], cp_full[:60] + 1.0)  # right side only
    write_case(tmp_path / "50.40" / "1_FRH26.80_RRH32.66_R-1.81_S17.77_CR7.19_Y-1.35_V9.48", left, cp_full[:60])

    monkeypatch.setattr(sys, "argv", ["load_rrf_sweeps.py", "--base", str(tmp_path), "--out-dir", str(tmp_path)])
    load_rrf_sweeps.main()

    train = np.load(tmp_path / "rrf_train.npz")
    test = np.load(tmp_path / "rrf_test.npz")

    assert train["Cp_matrix"].shape == (60, 2)
    assert test["Cp_matrix"].shape == (60, 1)
    assert np.all(train["y"] <= 0)
    np.testing.assert_array_equal(train["y"], test["y"])
    assert list(train["case_names"]) == [
        "1_FRH20.00_RRH35.00_R-0.84_S0.00_CR500.00_Y2.51_V19.44",
        "1_FRH40.00_RRH25.00_R1.50_S10.00_CR8.00_Y-1.00_V12.00",
    ]
    np.testing.assert_array_equal(train["R"], [-0.84, 1.5])
    np.testing.assert_array_equal(train["CR"], [500.0, 8.0])
    np.testing.assert_array_equal(test["V"], [9.48])
    # Same points as the target (after mirroring the right-only case), so Cp is copied unchanged.
    np.testing.assert_allclose(train["Cp_matrix"][:, 0], cp_full[:60])
    np.testing.assert_allclose(train["Cp_matrix"][:, 1], cp_full[:60] + 1.0)
    np.testing.assert_allclose(test["Cp_matrix"][:, 0], cp_full[:60])
    assert train["idw_spread_frac"].shape == (2,)
