"""Schematic of the car, the side diffuser and the attitude parameters.

Draws a not-to-scale explanatory figure for the report:
(a) top view: where the side diffusers are and which side carries sensors,
(b) side view: front and rear ride height (FRH, RRH) and pitch,
(c) front view: roll,
(d) top view in a corner: corner radius, steer and yaw.

Usage:  python experiments/vehicle_schematics.py --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arc, FancyArrowPatch
import numpy as np

INK, BODY, EDGE = "#1a1a19", "#dcdbd4", "#55554f"
DIFF, DIFF_EDGE, SENSOR, BLUE, ORANGE = "#cfe1f7", "#2a78d6", "#eb6834", "#2a78d6", "#eb6834"
GROUND = "#c9c8c0"


def rot(pts, ang, centre=(0.0, 0.0)):
    """Rotate an (n, 2) array of points by ang radians about centre."""
    c, s = np.cos(ang), np.sin(ang)
    p = np.asarray(pts, float) - centre
    return p @ np.array([[c, s], [-s, c]]) + centre


def rect(cx, cy, length, width, ang=0.0):
    """Corners of a rectangle centred on (cx, cy), long axis rotated by ang."""
    p = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                  [length / 2, width / 2], [-length / 2, width / 2]])
    return rot(p, ang) + (cx, cy)


def arrow(ax, a, b, color=INK, lw=1.2, style="-|>", ms=9, both=False, zorder=8):
    st = "<|-|>" if both else style
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=st, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def car_plan(ax, origin=(0, 0), ang=0.0, steer=0.0, scale=1.0, diffusers=True, sensors=False):
    """Top view of a simplified FSAE car. Nose at local x = 0, tail at x = 2.9, y up = car's right."""
    ox, oy = origin
    body = np.array([[0.0, -0.10], [0.55, -0.20], [0.85, -0.32], [2.30, -0.32], [2.85, -0.38],
                     [2.85, 0.38], [2.30, 0.32], [0.85, 0.32], [0.55, 0.20], [0.0, 0.10]])

    def place(p):
        return rot(np.asarray(p) * scale, ang) + (ox, oy)

    if diffusers:
        for sgn in (-1, 1):
            d = np.array([[1.0, 0.25 * sgn], [1.9, 0.25 * sgn], [1.9, 0.70 * sgn], [1.0, 0.70 * sgn]])
            ax.add_patch(Polygon(place(d), closed=True, fc=DIFF, ec=DIFF_EDGE, lw=1.2, zorder=2))
    ax.add_patch(Polygon(place(body), closed=True, fc=BODY, ec=EDGE, lw=1.2, zorder=3))
    ax.add_patch(Polygon(place([[1.25, -0.16], [1.85, -0.16], [1.85, 0.16], [1.25, 0.16]]),
                         closed=True, fc="#8a8980", ec=EDGE, lw=0.8, zorder=4))  # cockpit
    for wx, st in ((0.7, steer), (2.23, 0.0)):
        for sgn in (-1, 1):
            c = np.array([wx, 0.6 * sgn])
            w = rect(c[0], c[1], 0.46, 0.20, st)
            ax.add_patch(Polygon(place(w), closed=True, fc="#3b3b38", ec=INK, lw=0.8, zorder=4))
    if sensors:
        rng = np.random.default_rng(4)
        pts = np.column_stack([rng.uniform(1.07, 1.83, 14), -rng.uniform(0.30, 0.66, 14)])
        q = place(pts)
        ax.plot(q[:, 0], q[:, 1], "o", color=SENSOR, ms=4.2, mec="white", mew=0.6, zorder=6)


def panel_top(ax):
    car_plan(ax, sensors=True)
    for yy in (-0.45, 0.0, 0.45):
        arrow(ax, (-0.75, yy), (-0.25, yy), color=BLUE, lw=1.4)
    ax.text(-0.5, 0.62, "oncoming\nair", ha="center", fontsize=7.5, color=BLUE)
    ax.annotate("side diffuser, left side (y < 0):\nthe one that gets sensors (dots)", xy=(1.45, -0.62),
                xytext=(1.05, -1.22), fontsize=7.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color="#8a8980", lw=0.8))
    ax.annotate("right side: mirror image", xy=(1.45, 0.62), xytext=(1.45, 1.05), fontsize=7.5,
                color=INK, ha="center", arrowprops=dict(arrowstyle="-", color="#8a8980", lw=0.8))
    ax.text(0.7, -0.38, "front\nwheel", ha="center", va="top", fontsize=6.5, color=EDGE)
    ax.text(2.23, -0.38, "rear\nwheel", ha="center", va="top", fontsize=6.5, color=EDGE)
    # axes
    arrow(ax, (2.55, -1.1), (3.05, -1.1), lw=1.1)
    arrow(ax, (2.55, -1.1), (2.55, -0.65), lw=1.1)
    ax.text(3.08, -1.1, "x", va="center", fontsize=8)
    ax.text(2.55, -0.6, "y", ha="center", fontsize=8)
    ax.set(xlim=(-0.85, 3.25), ylim=(-1.45, 1.25), aspect="equal")
    ax.set_title("(a) Top view: where the side diffusers are", fontsize=8.8, loc="left", color=INK)


def panel_side(ax):
    frh, rrh = 0.10, 0.19                       # exaggerated
    pitch = np.arctan((rrh - frh) / (2.23 - 0.7))
    ax.fill_between([-0.3, 3.2], -0.06, 0, color=GROUND)
    ax.plot([-0.3, 3.2], [0, 0], color=INK, lw=1.6)
    r = 0.23
    # body silhouette with its floor on local y = 0, pitched about the front axle (rear goes up)
    base = np.array([[0.0, 0.12], [0.25, 0.0], [2.75, 0.0], [2.9, 0.12], [2.9, 0.50], [2.45, 0.55],
                     [2.05, 0.50], [1.82, 0.88], [1.70, 0.88], [1.50, 0.50], [0.95, 0.46], [0.0, 0.20]])
    body = rot(base, pitch, (0.7, 0.0)) + (0, frh)
    ax.add_patch(Polygon(body, closed=True, fc=BODY, ec=EDGE, lw=1.2, zorder=3))
    side = rot(np.array([[1.0, -0.005], [1.9, -0.005], [1.9, 0.09], [1.0, 0.05]]), pitch, (0.7, 0.0)) + (0, frh)
    ax.add_patch(Polygon(side, closed=True, fc=DIFF, ec=DIFF_EDGE, lw=1.2, zorder=5))
    for wx in (0.7, 2.23):
        ax.add_patch(Circle((wx, r), r, fc="#3b3b38", ec=INK, lw=0.8, zorder=6))
    for xa, ya, lab, dx, ha in ((0.7, frh, "FRH", -0.3, "right"), (2.23, rrh, "RRH", 0.3, "left")):
        arrow(ax, (xa, 0.0), (xa, ya), color=ORANGE, both=True, ms=6, lw=1.8)
        ax.plot([xa - 0.07, xa + 0.07], [ya, ya], color=ORANGE, lw=1.8, zorder=9)
        ax.annotate(lab, xy=(xa, ya * 0.6), xytext=(xa + dx, ya * 0.6), color=ORANGE, fontsize=8,
                    fontweight="bold", ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8), zorder=9)
    ax.text(1.45, -0.2, "front/rear ride height = floor-to-ground gap at the front/rear;\nthe side diffuser (blue) sits low, between the wheels",
            ha="center", va="top", fontsize=7.0, color=INK)
    ax.text(1.55, 1.18, "pitch: FRH and RRH differ\n(here nose down, as when braking)",
            ha="center", fontsize=7.2, color=INK)
    ax.text(-0.25, 0.75, "air →", fontsize=8, color=BLUE)
    ax.set(xlim=(-0.3, 3.25), ylim=(-0.5, 1.4), aspect="equal")
    ax.set_title("(b) Side view: ride heights and pitch", fontsize=8.8, loc="left", color=INK)


def panel_front(ax):
    phi = np.radians(5.0)                       # exaggerated roll
    ax.fill_between([-1.05, 1.05], -0.06, 0, color=GROUND)
    ax.plot([-1.05, 1.05], [0, 0], color=INK, lw=1.6)
    rc = (0.0, 0.12)
    chassis = np.array([[-0.30, 0.15], [0.30, 0.15], [0.30, 0.62], [0.18, 0.80], [-0.18, 0.80], [-0.30, 0.62]])
    ax.add_patch(Polygon(rot(chassis, phi, rc), closed=True, fc=BODY, ec=EDGE, lw=1.2, zorder=3))
    for sgn in (-1, 1):
        floor = np.array([[0.30 * sgn, 0.15], [0.72 * sgn, 0.15], [0.72 * sgn, 0.20], [0.30 * sgn, 0.20]])
        ax.add_patch(Polygon(rot(floor, phi, rc), closed=True, fc=DIFF, ec=DIFF_EDGE, lw=1.2, zorder=4))
        ax.add_patch(Polygon(rect(0.62 * sgn, 0.23, 0.18, 0.46), closed=True, fc="#3b3b38", ec=INK,
                             lw=0.8, zorder=6, alpha=0.92))
    for sgn, lab in ((-1, "smaller gap"), (1, "larger gap")):
        q = rot(np.array([[0.42 * sgn, 0.15]]), phi, rc)[0]
        arrow(ax, (q[0], 0.0), (q[0], q[1]), color=ORANGE, both=True, ms=5)
        ax.text(q[0], -0.09, lab, fontsize=6.8, color=ORANGE, ha="center", va="top")
    top = np.array(rc) + np.array([0.0, 1.0])
    ax.plot([rc[0], top[0]], [rc[1], top[1]], color="#8a8980", lw=0.9, ls="--", zorder=7)
    tilt = rot(np.array([top]), phi * 2.2, rc)[0]
    ax.plot([rc[0], tilt[0]], [rc[1], tilt[1]], color=INK, lw=1.0, zorder=7)
    ax.add_patch(Arc(rc, 1.7, 1.7, theta1=90, theta2=90 + np.degrees(phi * 2.2), color=INK, lw=1.0, zorder=7))
    ax.text(-0.22, 1.02, "roll φ", fontsize=8, color=INK, ha="right")
    ax.text(0.0, -0.27, "seen from the front: in a corner the body leans,\nso the floor on one side moves closer to the ground",
            ha="center", va="top", fontsize=7.0, color=INK)
    ax.set(xlim=(-1.1, 1.1), ylim=(-0.55, 1.2), aspect="equal")
    ax.set_title("(c) Front view: roll", fontsize=8.8, loc="left", color=INK)


def panel_corner(ax):
    R = 4.5
    th = np.linspace(np.radians(18), np.radians(122), 120)
    ax.plot(R * np.cos(th), R * np.sin(th), color="#8a8980", lw=1.2, ls="--")
    ax.plot(0, 0, "o", color=INK, ms=4)
    ax.text(-0.15, -0.12, "centre of the turn", fontsize=7, color=INK, va="top", ha="right")
    t0 = np.radians(58)
    p = np.array([R * np.cos(t0), R * np.sin(t0)])
    heading = t0 + np.pi / 2                    # travel direction (anticlockwise = left turn)
    yaw = np.radians(18)                        # nose points into the turn (exaggerated)
    sc = 0.75
    ang = heading + yaw + np.pi                 # car's local +x points backwards (nose at local x = 0)
    origin = p - rot(np.array([[1.45 * sc, 0.0]]), ang)[0]
    car_plan(ax, origin=origin, ang=ang, steer=np.radians(22), scale=sc, diffusers=True)
    ax.plot([0, p[0]], [0, p[1]], color=INK, lw=0.9, zorder=1)
    mid = 0.42 * p + 0.28 * np.array([np.sin(t0), -np.cos(t0)])
    ax.text(*mid, "corner radius R", fontsize=7.5, rotation=np.degrees(t0), ha="center", va="center", color=INK)
    v = np.array([np.cos(heading), np.sin(heading)])
    nose_dir = np.array([np.cos(heading + yaw), np.sin(heading + yaw)])
    arrow(ax, p, p + 3.6 * v, color=BLUE, lw=1.6)
    ax.plot(*np.array([p, p + 3.6 * nose_dir]).T, color=INK, lw=1.0, ls="--", zorder=7)
    ax.add_patch(Arc(p, 4.6, 4.6, theta1=np.degrees(heading), theta2=np.degrees(heading + yaw), color=INK, lw=1.0, zorder=7))
    tip_v, tip_n = p + 3.6 * v, p + 3.6 * nose_dir
    ax.text(tip_v[0] - 0.1, tip_v[1] + 0.2, "direction of travel (speed V)", fontsize=7, color=BLUE, ha="center")
    ax.text(tip_n[0] - 0.15, tip_n[1] + 0.05, "where the car points", fontsize=7, color=INK, ha="right", va="center")
    ym = p + 3.05 * np.array([np.cos(heading + yaw / 2), np.sin(heading + yaw / 2)])
    ax.text(ym[0], ym[1], "yaw", fontsize=7.5, color=INK, ha="center", va="center")
    fw = origin + rot(np.array([[0.7 * sc, -0.6 * sc]]), ang)[0]
    ax.annotate("steer: front wheels\nturned into the corner", xy=tuple(fw), xytext=(fw[0] - 2.9, fw[1] - 1.4),
                fontsize=7, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color="#8a8980", lw=0.8))
    ax.text(1.05, 0.05, "sideways acceleration = V\u00b2 / R\n(1.1\u20131.4 g in sweep 50.34)", fontsize=7.2, color=INK)
    ax.set(xlim=(-4.3, 5.2), ylim=(-0.7, 6.6), aspect="equal")
    ax.set_title("(d) Top view in a left-hand corner: radius, steer and yaw", fontsize=8.8, loc="left", color=INK)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", default="results/figures")
    out = Path(pa.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axs = plt.subplots(2, 2, figsize=(7.9, 5.5), gridspec_kw=dict(hspace=0.16, wspace=0.06))
    panel_top(axs[0, 0])
    panel_side(axs[0, 1])
    panel_front(axs[1, 0])
    panel_corner(axs[1, 1])
    for ax in axs.flat:
        ax.axis("off")
    fig.savefig(out / "fig_vehicle_schematic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
