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
| `side_diffuser_ensemble_dedup.npz` | experiments | 146,503 points × 100 cases of sweep 50.28; coordinates in mm |
| `case_params.csv` | experiments | one row per case (122 rows: sweep 50.28 then 50.34) |
| `loo_results.csv` | experiments | held-out folder and LOO RMS for each dedup column, from `check_correlation.py` |
| `rrf_train.npz`, `rrf_test.npz` | RRF validation | written by `cluster/load_rrf_sweeps.py` on the cluster |

**The dedup ensemble looks corrupted.** Its fields are almost flat (case 0 varies by 0.012 in Cp over the whole surface), and they do not match the highres ensemble of the same 100 cases. The likely cause is a unit mismatch: `all_points.csv` is in mm and `cluster/load_sweep_single_idw.py` does not convert the CFD points to the same unit. See `results/validation_extras_comparison.txt`. Prefer the highres ensemble until dedup is rebuilt.

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

## Experiments

Run from the repo root; results go to `results/`.

| Script | What it does |
|---|---|
| `experiments/validation_extras.py` | Leave-one-out (LOO) error, mean-field baseline, case spread, extrapolation test. With more params rows than cases, select them with `--rows "sweep == 50.28"` and check the order with `--case-order data/loo_results.csv`. |
| `experiments/placement_comparison.py` | LOO comparison of greedy, greedy with minimum spacing (0.02/0.05/0.10 m), QR pivoting and random layouts (20 seeds); writes a CSV and a bar chart. |
| `experiments/rrf_validation.py` | On `rrf_train.npz` / `rrf_test.npz`: modes for 90 % variance, training LOO, held-out test RMS and the mean-field baseline, for each placement method at 10, 20 and 24 sensors. |
| `experiments/time_core.py` | `perf_counter` timings of `compute_pod` and `greedy_sensor_placement` (40 sensors), with CPU model and NumPy version. |

Example:

```
python experiments/validation_extras.py data/side_diffuser_ensemble_highres.npz --params data/case_params.csv --rows "sweep == 50.28" --case-order data/loo_results.csv
python experiments/placement_comparison.py data/side_diffuser_ensemble_highres.npz --out results/placement_comparison_highres
python experiments/rrf_validation.py data/rrf_train.npz data/rrf_test.npz
```

`greedy_sensor_placement` takes `min_dist` (metres, default 0) and `coords` to keep sensors apart.

## Cluster scripts

These need numpy, pandas and scipy only.

- `cluster/master_load_sweep_ensemble.py`: loads sweeps, mirrors half-car cases (one-sided data on either side of y = 0) and aligns them with k=4 IDW.
- `cluster/load_rrf_sweeps.py`: builds `rrf_train.npz` (default sweeps 50.39 and 50.41) and `rrf_test.npz` (50.40). It keeps the left side, interpolates every case onto the first complete training case, lists skipped cases and reports how often the 4 IDW neighbours differ by more than 0.3 in Cp.

```
python cluster/load_rrf_sweeps.py --base /fs04/scratch2/he41/temp_runs/26_06 --out-dir .
```

## How to test

```
python -m pytest -v
```

The tests use synthetic data only, so they need no data files. `tests/test_core.py` checks:

- exact recovery of rank-1, rank-3 and rank-8 fields
- that the greedy selection returns unique indices
- that the vectorised greedy matches the original loop
- that no two picks are closer than `min_dist`, and that `min_dist=0` gives the unconstrained picks
- that the reconstruction error falls as sensors are added under noise
- that NaN input raises `CfdDataError`

`tests/test_loaders.py` checks half-car mirroring on each side, and `tests/test_load_rrf_sweeps.py` runs the RRF loader on a small synthetic sweep tree.

The latest report is in `docs/test_report.txt`.

API docs are in `docs/html/`. To regenerate them:

```
python -m pdoc -o docs/html sensor_opt
```
