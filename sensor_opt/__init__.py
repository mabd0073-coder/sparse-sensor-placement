"""Sparse pressure-sensor placement on a side diffuser: POD, greedy selection, reconstruction."""

from sensor_opt.core import CfdDataError, compute_pod, greedy_sensor_placement, reconstruct

__all__ = ["CfdDataError", "compute_pod", "greedy_sensor_placement", "reconstruct"]
