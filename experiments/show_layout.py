"""Interactive 3-D view of a sensor layout on the side diffuser (opens in a browser).

Draws the left-side surface coloured by the mean Cp of the ensemble (or grey),
the CSV layout as numbered markers, and optionally the unconstrained greedy
layout for comparison. Hover over a sensor to see its coordinates, edge
distance and nearest-sensor spacing.

Usage:
    python experiments/show_layout.py data/side_diffuser_ensemble_highres_NN.npz \
        --layout results/feasible_layout.csv -o results/feasible_layout_3d.html
"""
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--layout", default="results/feasible_layout.csv")
    p.add_argument("--points", type=int, default=60000, help="surface points to draw")
    p.add_argument("-o", "--out", default="results/feasible_layout_3d.html")
    a = p.parse_args()

    z = np.load(a.ensemble)
    xyz = np.column_stack([z["x"], z["y"], z["z"]]).astype(float)
    if np.ptp(xyz, axis=0).max() > 20:
        xyz = xyz / 1000.0
    keep = xyz[:, 1] <= 0
    cp = z["Cp_matrix"][keep].mean(axis=1)
    xyz = xyz[keep]
    sub = np.random.default_rng(0).choice(len(xyz), min(a.points, len(xyz)), replace=False)

    L = pd.read_csv(a.layout)
    sx, sy, sz = L.x_mm / 1000, L.y_mm / 1000, L.z_mm / 1000
    hover = [f"sensor {s}<br>x {x:.0f}, y {y:.0f}, z {zz:.0f} mm<br>edge {e:.0f} mm, nearest sensor {n:.0f} mm"
             for s, x, y, zz, e, n in zip(L.sensor, L.x_mm, L.y_mm, L.z_mm, L.edge_distance_mm, L.nearest_sensor_mm)]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xyz[sub, 0], y=xyz[sub, 1], z=xyz[sub, 2], mode="markers",
                               marker=dict(size=1.5, color=cp[sub], colorscale="Jet", cmin=-2, cmax=1,
                                           colorbar=dict(title="mean Cp")), hoverinfo="skip", name="surface"))
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers+text", text=[str(s) for s in L.sensor],
                               textposition="top center", hovertext=hover, hoverinfo="text",
                               marker=dict(size=6, color="white", line=dict(color="black", width=2)),
                               name="sensors"))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
                      title=f"{len(L)} sensors from {a.layout}", margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(a.out)
    print(f"Saved {a.out}; open it in a browser")


if __name__ == "__main__":
    main()
