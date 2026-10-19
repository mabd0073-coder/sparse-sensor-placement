"""Space-filling CFD sweep designs for the side-diffuser POD/SVD ensemble.

The older sweeps (generate_sweeps.py, 50.34) move every parameter together
with np.linspace, so each sweep is a single line through parameter space and
POD cannot separate the effect of one parameter from another. This script
uses scrambled Sobol sampling so that ride height, roll, yaw and corner
radius vary independently and fill the operating envelope.

All runs are RRF so the whole new ensemble shares one simulation frame:
  * cornering:  corner radius 4-18 m, steer and speed taken from the team's
                limit-cornering curve (steer2moreres_50.34.csv) so the cases
                are consistent with sweep 50.34.
  * straight:   corner radius 500 m (Mark's workaround: RRF that is
                effectively straight, and keeps side_diffuser.csv).

Output files (team format: case,frh,rrh,r,s,cr,y,v,run), one per sweep:
  new/cornering_rrf_batch1.csv   32 cases  (priority 1)
  new/straight_rrf_batch1.csv    32 cases  (priority 2)
  new/heldout_rrf_test.csv       16 cases  (8 cornering + 8 straight, never train on these)
  new/cornering_rrf_batch2.csv   32 cases  (optional, extends the same Sobol sequence)
  new/straight_rrf_batch2.csv    32 cases  (optional)

Edit the constants below and re-run to change any range.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

HERE = Path(__file__).parent
OUT = HERE / "new"
STEER_CURVE = HERE / "steer2moreres_50.34.csv"  # team case file for sweep 50.34

# ---- Ranges (confirm with Lachlan/Mark before submitting) -----------------
FRH = (15.0, 55.0)          # mm, same box as 50.28 (10x10 grid, 15-55 mm)
RRH = (15.0, 55.0)          # mm
ROLL_MAX = 3.0              # deg, range of roll sweep 50.27
YAW_STRAIGHT = (-6.0, 6.0)  # deg
YAW_CORNER = (-4.0, 4.0)    # deg, body slip in a corner
CR_CORNER = (4.0, 18.0)     # m, range of 50.34
CR_STRAIGHT = 500.0         # m, effectively straight (RRF workaround)
V_STRAIGHT = 19.44          # m/s, baseline speed used for Cp in analysis.py

# Ground-clearance guard. Low ride height combined with large roll has never
# been run (50.28: roll 0 down to 15 mm; 50.27: roll +-3 deg at 31.5/32 mm).
# Allowed |roll| therefore scales from 0 at RH_MIN to ROLL_MAX at RH_FULL_ROLL.
RH_MIN = 15.0
RH_FULL_ROLL = 31.5

N_BATCH = 32                # Sobol batches are powers of two
N_TEST_EACH = 8
SEED_TRAIN = 3001
SEED_TEST = 2026

COLS = ["case", "frh", "rrh", "r", "s", "cr", "y", "v", "run"]


def scale(u, lo_hi):
    lo, hi = lo_hi
    return lo + u * (hi - lo)


def roll_limit(frh, rrh):
    frac = (np.minimum(frh, rrh) - RH_MIN) / (RH_FULL_ROLL - RH_MIN)
    return ROLL_MAX * np.clip(frac, 0.0, 1.0)


def cornering_cases(u, curve):
    """u: (n, 5) unit samples -> cr, frh, rrh, roll fraction, yaw."""
    cr = scale(u[:, 0], CR_CORNER)
    frh = scale(u[:, 1], FRH)
    rrh = scale(u[:, 2], RRH)
    r = (2 * u[:, 3] - 1) * roll_limit(frh, rrh)
    y = scale(u[:, 4], YAW_CORNER)
    s = np.interp(cr, curve["cr"], curve["s"])
    v = np.interp(cr, curve["cr"], curve["v"])
    return pd.DataFrame(dict(frh=frh, rrh=rrh, r=r, s=s, cr=cr, y=y, v=v))


def straight_cases(u):
    """u: (n, 4) unit samples -> frh, rrh, roll fraction, yaw."""
    frh = scale(u[:, 0], FRH)
    rrh = scale(u[:, 1], RRH)
    r = (2 * u[:, 2] - 1) * roll_limit(frh, rrh)
    y = scale(u[:, 3], YAW_STRAIGHT)
    n = len(u)
    return pd.DataFrame(dict(frh=frh, rrh=rrh, r=r, s=np.zeros(n),
                             cr=np.full(n, CR_STRAIGHT), y=y,
                             v=np.full(n, V_STRAIGHT)))


def finalise(df):
    df = df.round(2).reset_index(drop=True)
    df = df + 0.0  # turn -0.0 into 0.0
    if df.duplicated().any():
        raise ValueError("duplicate cases after rounding")
    df.insert(0, "case", np.arange(1, len(df) + 1))
    df["run"] = True
    return df[COLS]


def main():
    curve = pd.read_csv(STEER_CURVE).sort_values("cr")
    OUT.mkdir(exist_ok=True)

    # Two consecutive Sobol batches: batch1 alone is balanced, batch1+batch2 more so.
    u_c = qmc.Sobol(d=5, scramble=True, seed=SEED_TRAIN).random_base2(m=6)
    u_s = qmc.Sobol(d=4, scramble=True, seed=SEED_TRAIN + 1).random_base2(m=6)

    # Held-out test points: Latin hypercube with a different seed.
    t_c = qmc.LatinHypercube(d=5, seed=SEED_TEST).random(N_TEST_EACH)
    t_s = qmc.LatinHypercube(d=4, seed=SEED_TEST + 1).random(N_TEST_EACH)

    files = {
        "cornering_rrf_batch1.csv": cornering_cases(u_c[:N_BATCH], curve),
        "straight_rrf_batch1.csv": straight_cases(u_s[:N_BATCH]),
        "heldout_rrf_test.csv": pd.concat([cornering_cases(t_c, curve),
                                           straight_cases(t_s)]),
        "cornering_rrf_batch2.csv": cornering_cases(u_c[N_BATCH:], curve),
        "straight_rrf_batch2.csv": straight_cases(u_s[N_BATCH:]),
    }
    for name, df in files.items():
        df = finalise(df)
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {len(df)} cases")


if __name__ == "__main__":
    main()
