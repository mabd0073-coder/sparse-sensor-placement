# AI-assisted changes log

Changes made with Claude Code (Claude Opus 5.5), one line per change.

- 2026-10-25 `9dc6a2c` analysis.py loads the ensemble and exclusion mask from `data/`, overridable with `SSP_DATA_DIR` and `SSP_ENSEMBLE`.
- 2026-10-25 `229159d` Vectorised the greedy sensor placement; it picks the same sensors as the old loop (checked on real modes).
- 2026-10-25 `fac8f4d` The exclusion mask is applied only when it can be aligned to the mesh; otherwise a warning is printed and no mask is used.
- 2026-10-25 `d9388e2` analysis.py also reports a held-out RMS error (modes and sensors rebuilt without the test case).
- 2026-10-25 `697fa75` Removed the unused numpy alias patch and duplicate prints; fixed the plot title's sensor count; error plots use a 0–0.5 colour range.
- 2026-10-25 `0fc4a9f` Added `sensor_opt/core.py` with `compute_pod`, `greedy_sensor_placement`, `reconstruct` and `CfdDataError`, with NumPy-style docstrings.
- 2026-10-25 `d921c60` analysis.py imports the core functions instead of defining them; output on the highres ensemble is identical to the previous version.
- 2026-10-25 `a4a90c2` analysis.py runs its pipeline only under `if __name__ == "__main__":`.
- 2026-10-25 `d6b1bf1` Added `tests/test_core.py` (10 pytest tests) and `pytest.ini`.
- 2026-10-25 `eaa0f62` Added `docs/test_report.txt` from `pytest -v` (10 passed).
- 2026-10-25 `8495c23` Added pdoc HTML docs in `docs/html/` and un-ignored them in `.gitignore`.
- 2026-10-25 `e2204ef` Rewrote the README to cover purpose, install, data location, how to run and how to test.
- 2026-10-25 Added this log.
- 2026-10-25 Merged feat/tests-docs into feat/final-analysis (base docs/report-figures lacked sensor_opt, tests and docs).
- 2026-10-25 `greedy_sensor_placement` takes an optional `min_dist` (metres, default 0 = unchanged picks); tests for spacing and for identical picks at 0.
- 2026-10-25 Fixed half-car detection in cluster/load_sweep_ensemble.py and master_load_sweep_ensemble.py so one-sided data on either side is mirrored; added tests/test_loaders.py.
- 2026-10-25 Added cluster/load_rrf_sweeps.py (RRF train/test ensembles on left-side points, k=4 IDW, neighbour-spread diagnostic) and tests/test_load_rrf_sweeps.py.
- 2026-10-25 validation_extras.py: --rows (pandas query) and --case-order checks to match case_params rows to ensemble columns, name matching when the npz stores case_names; reran on dedup and highres (results/validation_extras_*.txt).
- 2026-10-25 Added experiments/placement_comparison.py (LOO: greedy, greedy+min_dist 0.02/0.05/0.10 m, QR, random x20, mean-field baseline); results on dedup in results/placement_comparison*; results/*.csv and *.png un-ignored.
- 2026-10-25 Added experiments/rrf_validation.py (modes, training LOO, held-out test RMS and mean-field baseline for greedy, greedy+min_dist, QR, random at 10/20/24 sensors); smoke-tested on a highres split, awaiting real rrf_*.npz.
- 2026-10-25 Ran placement_comparison.py on the highres ensemble (results/placement_comparison_highres*).
- 2026-10-25 Profiled analysis.py with cProfile (results/profile_analysis.txt, top 25 rows) and added experiments/time_core.py; laptop timings in results/timings_laptop.txt.
- 2026-10-25 Regenerated docs/test_report.txt (20 tests).
- 2026-10-25 Regenerated docs/html (min_dist docs).
- 2026-10-25 README: data files, dedup warning, experiments, cluster scripts and new tests.
- 2026-10-25 Merged PR #1 (feat/final-analysis) and docs/report-figures into main; regenerated docs/test_report.txt (20 passed) and docs/html on main.
- 2026-10-25 Merged docs/report-figures into main (feasible_layout.py, show_layout.py, pipeline_figure.py, one-column figures, results/feasible_layout*); reproduced results/feasible_layout.csv byte for byte.
- 2026-10-25 Added tests/test_feasible_layout.py (greedy keeps to the allowed mask and min_dist; exchange never lowers log det).
- 2026-10-25 README: Sensor layout section (rebuild and 3-D view commands); dedup ensemble (A0) withdrawn. Added report/README.md placeholder for the final PDF.
- 2026-10-25 Regenerated docs/test_report.txt (22 passed) and docs/html.
