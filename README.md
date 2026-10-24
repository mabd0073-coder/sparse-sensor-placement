# Sparse Sensor Placement

Optimal pressure-sensor placement on a side diffuser using POD and greedy D-optimal selection, validated on CFD sweep data. MMA3001 project.

## Purpose

Given an ensemble of CFD surface-pressure (CpS) fields from a ride-height sweep, the pipeline:

1. builds a reduced POD basis that captures 90% of the pressure variance,
2. greedily picks 40 sensor locations on one side of the diffuser that make that basis best conditioned, and
3. reconstructs the full pressure field from those 40 readings and reports the error, both in-sample and on a held-out case.

The reusable maths lives in the `sensor_opt` package (`compute_pod`, `greedy_sensor_placement`, `reconstruct`). `analysis.py` loads the data, applies the candidate masks, runs the pipeline and makes the plots.

## Install

Python 3.12 is used for development.

```
python -m pip install numpy scipy pandas matplotlib plotly
python -m pip install pytest pdoc        # for tests and API docs
```

## Data location

The data is not in git. Put the files in `data/` at the repo root:

| File | Needed | Contents |
|---|---|---|
| `side_diffuser_ensemble_highres.npz` | yes (default) | `Cp_matrix` (points × cases), `x`, `y`, `z` |
| `side_diffuser_ensemble_highres_NN.npz` | optional | same layout, alternative ensemble |
| `exclusion_mask.npz` | optional | `excluded` boolean per point, optionally `x`, `y`, `z` |

Two environment variables override the defaults:

- `SSP_DATA_DIR`: the data folder (default `data`)
- `SSP_ENSEMBLE`: the ensemble file name within that folder

The current `exclusion_mask.npz` was built for an older mesh and has no coordinates. `analysis.py` detects this, prints a warning and runs without the mask, so edge points can be selected as sensors. Regenerate the mask for the current mesh to restore edge exclusion.

## How to run

From the repo root:

```
python analysis.py
```

To use the alternative ensemble (PowerShell):

```
$env:SSP_ENSEMBLE = "side_diffuser_ensemble_highres_NN.npz"; python analysis.py; Remove-Item Env:SSP_ENSEMBLE
```

The script prints the selected sensors and the in-sample and held-out RMS errors. It writes PNG plots and `reconstructed_3d.html` to the repo root, overwriting the previous run's files. On Windows, close any open preview tabs of those PNGs first; a file held open by another program makes the save fail with `OSError: [Errno 22]`.

## How to test

```
python -m pytest -v
```

The tests in `tests/test_core.py` use synthetic data only, so they need no data files. They check:

- exact recovery of rank-1, rank-3 and rank-8 fields
- that the greedy selection returns unique indices
- that the vectorised greedy matches the original loop
- that the reconstruction error falls as sensors are added under noise
- that NaN input raises `CfdDataError`

The latest report is in `docs/test_report.txt`.

API docs are in `docs/html/`. To regenerate them:

```
python -m pdoc -o docs/html sensor_opt
```
