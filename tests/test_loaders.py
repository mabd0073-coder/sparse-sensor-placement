import numpy as np
import pandas as pd
import pytest

from cluster import load_sweep_ensemble, master_load_sweep_ensemble

LOADERS = [load_sweep_ensemble, master_load_sweep_ensemble]


def one_sided_case(y_low, y_high, n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.uniform(0.2, 1.2, n),
        "y": rng.uniform(y_low, y_high, n),
        "z": rng.uniform(0.0, 0.2, n),
        "CpS": rng.standard_normal(n),
    })


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("y_range", [(-0.70, -0.23), (0.23, 0.70)], ids=["left", "right"])
def test_one_sided_data_is_mirrored(loader, y_range):
    df = one_sided_case(*y_range)

    coords, cp, mirrored = loader.prepare_full_car_data(df)

    assert mirrored
    assert len(coords) == 2 * len(df)
    np.testing.assert_array_equal(coords[len(df):, 1], -df["y"].to_numpy())
    np.testing.assert_array_equal(cp, np.concatenate([df["CpS"], df["CpS"]]))


@pytest.mark.parametrize("loader", LOADERS)
def test_full_car_data_is_not_mirrored(loader):
    df = one_sided_case(-0.70, 0.70)

    coords, cp, mirrored = loader.prepare_full_car_data(df)

    assert not mirrored
    assert len(coords) == len(df)
