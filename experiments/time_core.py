"""Wall-clock timings of compute_pod and greedy_sensor_placement on a real ensemble.

Each function is timed with time.perf_counter over --repeats runs, with the
settings analysis.py uses (90 % variance, 2-mode minimum) on every point of the
given ensemble (analysis.py first drops duplicate points).
The report includes the CPU model and the NumPy version.

Usage:
    python experiments/time_core.py data/side_diffuser_ensemble_highres.npz > results/timings_laptop.txt
"""
import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensor_opt.core import compute_pod, greedy_sensor_placement  # noqa: E402


def cpu_model():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def timed(fn, repeats):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return times, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--key", default="Cp_matrix")
    p.add_argument("--sensors", type=int, default=40)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()

    X = np.load(a.ensemble)[a.key]
    print(f"CPU: {cpu_model()} ({os.cpu_count()} logical cores)")
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, {platform.platform()}")
    print(f"Ensemble: {a.ensemble}, {X.shape[0]} points x {X.shape[1]} cases")

    t_pod, (U_r, _, r) = timed(lambda: compute_pod(X, 0.90), a.repeats)
    t_greedy, _ = timed(lambda: greedy_sensor_placement(U_r, a.sensors), a.repeats)

    print(f"\n{'function':40s} " + " ".join(f"run {i + 1:>2d} [s]" for i in range(a.repeats)) + "   best [s]   mean [s]")
    for name, t in ((f"compute_pod (r = {r} modes)", t_pod),
                    (f"greedy_sensor_placement ({a.sensors} sensors)", t_greedy)):
        print(f"{name:40s} " + " ".join(f"{v:10.3f}" for v in t) + f" {min(t):10.3f} {np.mean(t):10.3f}")


if __name__ == "__main__":
    main()
