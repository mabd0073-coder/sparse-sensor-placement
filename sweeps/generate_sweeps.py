import numpy as np
import pandas as pd

def create_sweep_csv(filename, n_cases, r_range, s_range, cr_range, y_range, v_range, frh=31.5, rrh=32.0):
    df = pd.DataFrame({
        'case': range(1, n_cases + 1),
        'frh': frh,
        'rrh': rrh,
        'r': np.round(np.linspace(r_range[0], r_range[1], n_cases), 4),
        's': np.round(np.linspace(s_range[0], s_range[1], n_cases), 4),
        'cr': np.round(np.linspace(cr_range[0], cr_range[1], n_cases), 4),
        'y': np.round(np.linspace(y_range[0], y_range[1], n_cases), 4),
        'v': np.round(np.linspace(v_range[0], v_range[1], n_cases), 4),
        'run': True
    })
    df.to_csv(filename, index=False)
    print(f"Saved {filename} with {n_cases} cases.")

# Generate 50-case sweeps
create_sweep_csv("yaw_sweep_50.csv", 50, r_range=(0,0), s_range=(0,0), cr_range=(0,0), y_range=(0, 15), v_range=(15, 15))
create_sweep_csv("roll_sweep_50.csv", 50, r_range=(-3, 3), s_range=(0,0), cr_range=(0,0), y_range=(0, 0), v_range=(15, 15))
create_sweep_csv("steer_cr_vel_sweep_50.csv", 50, r_range=(0,0), s_range=(28.5, 6.0), cr_range=(4.0, 18.0), y_range=(0, 0), v_range=(6.5, 15.5))

# Generate 100-case sweeps (if you want maximum density)
create_sweep_csv("yaw_sweep_100.csv", 100, r_range=(0,0), s_range=(0,0), cr_range=(0,0), y_range=(0, 15), v_range=(15, 15))
create_sweep_csv("roll_sweep_100.csv", 100, r_range=(-3, 3), s_range=(0,0), cr_range=(0,0), y_range=(0, 0), v_range=(15, 15))
create_sweep_csv("steer_cr_vel_sweep_100.csv", 100, r_range=(0,0), s_range=(28.5, 6.0), cr_range=(4.0, 18.0), y_range=(0, 0), v_range=(6.5, 15.5))