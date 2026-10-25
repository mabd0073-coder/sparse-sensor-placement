"""Toy model used in the report to explain the physics and the method.

This is NOT the team's geometry and NOT CFD. It is a quasi-one-dimensional
"venturi" model of the air gap under a car floor, used only to show a reader
without an aerodynamics background

1. why ride height and pitch change the pressure under the floor, and
2. how the POD/SVD + greedy sensor + least-squares method works, step by step.

Model
-----
Air flows front-to-rear through the gap h(x) between the floor and the ground.
Continuity (incompressible, per unit width) gives V(x) h(x) = V_e h_e, and if
the air leaves the diffuser exit at free-stream pressure, Bernoulli gives

    Cp(x) = 1 - (V(x)/V_inf)^2 = 1 - (h_e / h(x))^2 .

The gap is the local ride height (a straight line between the front and rear
ride heights, FRH and RRH) plus a fixed offset and the floor shape (a raised
inlet lip and a diffuser ramp at the rear). The model ignores friction, flow
separation, 3-D leakage round the floor edges and the car's other parts, so
it is only qualitative.

Outputs (PNG): fig_diffuser_model.png, fig_method_toy.png, and a printed
summary of the numbers quoted in the report.

Usage:  python experiments/toy_diffuser_demo.py --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- toy geometry (illustrative values only) --------------------------------
X0 = 0.30      # m, floor starts this far behind the front axle
L = 0.90       # m, floor length
WB = 1.53      # m, wheelbase (FRH measured at front axle, RRH at rear axle)
H0 = 20.0      # mm, floor sits this far above the ride-height reference
DELTA = 15.0   # mm, diffuser ramp rise at the exit
LIP, LIP_LEN = 12.0, 0.10   # mm, m: raised inlet lip
X_T = 0.45     # m, start of the diffuser ramp
x = np.linspace(0.0, L, 240)

INK, GREY, GRID = "#1a1a19", "#a3a29b", "#e8e7e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def floor_offset(xx):
    """Floor shape above the local ride height [mm]: inlet lip + diffuser ramp."""
    d = np.zeros_like(xx)
    m = xx < LIP_LEN
    d[m] = LIP * (1 - xx[m] / LIP_LEN)
    s = np.clip((xx - X_T) / (L - X_T), 0, 1)
    return d + DELTA * (1 - np.cos(np.pi * s)) / 2


def gap(frh, rrh):
    """Gap between floor and ground [mm] along the floor."""
    return frh + (rrh - frh) * (X0 + x) / WB + H0 + floor_offset(x)


def cp(frh, rrh):
    """Pressure coefficient along the floor from continuity + Bernoulli."""
    h = gap(frh, rrh)
    return 1.0 - (h[-1] / h) ** 2


def greedy(U_r, k, eps=1e-10):
    """Greedy D-optimal sensor selection (same algorithm as analysis.py)."""
    n, r = U_r.shape
    sel, avail, D = [], np.ones(n, bool), np.eye(r) / eps
    for _ in range(k):
        g = 1 + ((U_r @ D) * U_r).sum(1)
        g[~avail] = -np.inf
        b = int(np.argmax(g))
        sel.append(b)
        avail[b] = False
        w = U_r[b][:, None]
        Dw = D @ w
        D -= Dw @ Dw.T / (1 + (w.T @ Dw).item())
    return sel


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8980")
    ax.tick_params(colors="#55554f", labelsize=8)


def title(ax, text):
    ax.set_title(text, fontsize=8.8, loc="left", color=INK)


def fig_physics(out, column=False):
    if column:   # one journal column: panels stacked
        fig = plt.figure(figsize=(3.45, 5.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1, 1.05], hspace=0.95)
    else:
        fig = plt.figure(figsize=(8.2, 3.0))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 0.95], wspace=0.42)

    # (a) geometry with speed arrows (arrow length proportional to air speed)
    ax = fig.add_subplot(gs[0])
    h = gap(30, 30)
    ax.fill_between(x, 0, h, color="#eef4fc")
    ax.fill_between(x, h, h + 7, color="#c9c8c0")
    ax.plot(x, h, color=INK, lw=2.2)
    ax.axhline(0, color=INK, lw=2.2)
    ax.fill_between(x, -6, 0, color="#d8d7d0", lw=0)
    for xi in (0.13, 0.33, 0.55, 0.74):
        i = np.argmin(abs(x - xi))
        v = h[-1] / h[i]                       # speed relative to the exit speed
        ax.annotate("", xy=(xi + 0.11 * v, 12), xytext=(xi, 12),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8, mutation_scale=10))
    ax.text(0.30, 61, "car floor", fontsize=8, color=INK, ha="center")
    if column:
        ax.text(0.27, 22, "narrow gap: fast air,\nlow pressure", fontsize=6.3, color=INK, ha="center")
        ax.text(0.72, 22, "gap opens: air\nslows, pressure\nrecovers", fontsize=6.3, color=INK, ha="center")
    else:
        ax.text(0.27, 26, "narrow gap:\nfast air,\nlow pressure", fontsize=7, color=INK, ha="center")
        ax.text(0.73, 26, "gap opens:\nair slows,\npressure\nrecovers", fontsize=7, color=INK, ha="center")
    ax.text(0.45, -15, "ground", fontsize=7.5, color="#55554f", ha="center")
    ax.text(0.0, 82, "air flow →", fontsize=8, color=BLUE)
    ax.set(xlim=(-0.02, 0.92), ylim=(-19, 90), xlabel="distance along floor, front → rear [m]",
           ylabel="gap to ground [mm]")
    title(ax, "(a) Air gap under the floor\narrows ∝ air speed; not to scale")
    if column:
        ax.set_xlabel("distance along floor [m]")
    style(ax)

    # (b) Cp along the floor for different attitudes
    ax = fig.add_subplot(gs[1])
    cases = [((15, 15), "low, level: 15/15", BLUE, "-"),
             ((55, 55), "high, level: 55/55", AQUA, "-"),
             ((15, 55), "nose down: 15/55", ORANGE, "-"),
             ((55, 15), "nose up: 55/15", YELLOW, "--")]
    for (f, r), lab, c, ls in cases:
        ax.plot(x, cp(f, r), color=c, lw=2, ls=ls, label=lab)
    ax.axhline(0, color="#8a8980", lw=0.8)
    if not column:
        ax.text(0.9, 0.1, "Cp = 0: same as\nsurrounding air", fontsize=6.8, ha="right", color="#55554f")
    ax.set(xlabel="distance along floor [m]", ylabel="pressure coefficient Cp")
    title(ax, "(b) Pressure along the floor\nmore negative = more suction")
    ax.legend(frameon=False, fontsize=5.6 if column else 6.6, loc="lower right", title="FRH/RRH [mm]",
              title_fontsize=5.6 if column else 6.6, handlelength=1.6, borderaxespad=0.1)
    style(ax)

    # (c) aeromap: mean suction over FRH x RRH, with the 10 x 10 sweep grid
    ax = fig.add_subplot(gs[2])
    fr = np.linspace(13, 57, 89)
    F, R = np.meshgrid(fr, fr, indexing="ij")
    S = np.vectorize(lambda a, b: -cp(a, b).mean())(F, R)
    cs = ax.contourf(F, R, S, levels=12, cmap="Blues")
    g = np.linspace(15, 55, 10)
    G1, G2 = np.meshgrid(g, g)
    ax.plot(G1, G2, "+", color=INK, ms=4, mew=0.8)
    cb = fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("mean suction (−Cp)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set(xlabel="front ride height FRH [mm]", ylabel="rear ride height RRH [mm]",
           xlim=(13, 57), ylim=(13, 57))
    title(ax, "(c) Downforce 'aeromap'\n+ = sweep 50.28 cases")
    style(ax)
    ax.grid(False)

    fig.savefig(out / ("fig_diffuser_model_col.png" if column else "fig_diffuser_model.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_method(out, column=False):
    g = np.linspace(15, 55, 10)
    cases = [(a, b) for a in g for b in g]
    X = np.column_stack([cp(a, b) for a, b in cases])
    mean = X.mean(1, keepdims=True)
    U, s, _ = np.linalg.svd(X - mean, full_matrices=False)
    energy = s ** 2 / np.sum(s ** 2)

    test = (22.0, 48.0)
    truth = cp(*test)
    r = 2
    sel = greedy(U[:, :r], r)
    a, *_ = np.linalg.lstsq(U[sel, :r], truth[sel] - mean[sel, 0], rcond=None)
    rec = mean[:, 0] + U[:, :r] @ a
    rms_rec = np.sqrt(np.mean((rec - truth) ** 2))
    rms_base = np.sqrt(np.mean((mean[:, 0] - truth) ** 2))

    if column:   # one journal column: 2 x 2 with smaller type
        plt.rcParams.update({"font.size": 6.5})
        fig = plt.figure(figsize=(3.45, 3.9))
        gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.75)
    else:
        fig = plt.figure(figsize=(7.6, 5.6))
        gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    for j in range(X.shape[1]):
        ax.plot(x, X[:, j], color=GREY, lw=0.5, alpha=0.6)
    ax.plot(x, mean[:, 0], color=INK, lw=2.2, label="mean of all 100 cases")
    ax.set(xlabel="position along floor [m]", ylabel="Cp")
    title(ax, "(a) 100 toy cases: one Cp curve each")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(ax)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(X, aspect="auto", cmap="Blues_r", extent=[0.5, 100.5, L, 0])
    ax.set(xlabel="case number (column of X)", ylabel="position [m] (row of X)")
    title(ax, "(b) Same data as a matrix: 1 column per case")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Cp", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    style(ax)
    ax.grid(False)

    ax = fig.add_subplot(gs[1, 0])
    cols = [BLUE, ORANGE, AQUA]
    for i in range(3):
        m = U[:, i] / np.max(np.abs(U[:, i]))
        ax.plot(x, m, color=cols[i], lw=2, label=f"mode {i+1}: {energy[i]*100:.{1 if energy[i] > 0.01 else 2}f} %")
    ax.axhline(0, color="#8a8980", lw=0.8)
    ax.set(xlabel="position along floor [m]", ylabel="mode shape (scaled)", ylim=(-1.15, 2.0))
    title(ax, "(c) SVD: the patterns (% of variation)")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right", ncol=1)
    style(ax)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, truth, color=INK, lw=2.6, label="true field (unseen case)")
    ax.plot(x, mean[:, 0], color=GREY, lw=1.6, ls="--", label=f"guess = average field (RMS {rms_base:.2f})")
    ax.plot(x, rec, color=BLUE, lw=1.8, ls=(0, (4, 2)), label=f"2 sensors + 2 modes (RMS {rms_rec:.3f})")
    ax.plot(x[sel], truth[sel], "o", color=ORANGE, ms=8, mec="white", mew=1.2, label="sensor readings", zorder=5)
    ax.set(xlabel="position along floor [m]", ylabel="Cp", ylim=(-3.0, 0.1))
    title(ax, f"(d) Rebuild an unseen case ({test[0]:.0f}/{test[1]:.0f} mm)")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    style(ax)

    if column:
        short = ["(a) 100 toy cases", "(b) as a matrix X", "(c) SVD patterns", "(d) rebuild unseen case"]
        data_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        for ax, t in zip(data_axes, short):
            ax.set_title(t, loc="left")
        leg = data_axes[3].get_legend()
        h, l = data_axes[3].get_legend_handles_labels()
        leg.remove()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=5.2, bbox_to_anchor=(0.5, -0.06))
        data_axes[0].get_legend().remove()
        for ax in fig.axes:
            ax.title.set_fontsize(6.6)
            ax.tick_params(labelsize=5.8)
            ax.xaxis.label.set_fontsize(6)
            ax.yaxis.label.set_fontsize(6)
            leg = ax.get_legend()
            if leg:
                for t in leg.get_texts():
                    t.set_fontsize(5.2)
        fig.savefig(out / "fig_method_toy_col.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        plt.rcParams.update({"font.size": 8.5})
        return
    fig.savefig(out / "fig_method_toy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # numbers quoted in the report
    loo, base, rs = [], [], []
    for j in range(X.shape[1]):
        tr = np.delete(X, j, 1)
        m = tr.mean(1, keepdims=True)
        Uu, ss, _ = np.linalg.svd(tr - m, full_matrices=False)
        rr = int(np.searchsorted(np.cumsum(ss ** 2) / np.sum(ss ** 2), 0.90) + 1)
        rs.append(rr)
        sl = greedy(Uu[:, :rr], 10)
        aa, *_ = np.linalg.lstsq(Uu[sl, :rr], X[sl, j] - m[sl, 0], rcond=None)
        loo.append(np.sqrt(np.mean((m[:, 0] + Uu[:, :rr] @ aa - X[:, j]) ** 2)))
        base.append(np.sqrt(np.mean((m[:, 0] - X[:, j]) ** 2)))
    clustered = greedy(U[:, :1], 3)
    print(f"energy of modes 1-3 [%]: {np.round(energy[:3]*100, 3)}")
    print(f"Cp range in toy ensemble: {X.min():.2f} to {X.max():.2f}")
    print(f"test case {test}: mean-field RMS {rms_base:.3f}, 2-sensor/2-mode RMS {rms_rec:.4f}, "
          f"sensors at x = {np.round(x[sel], 3)} m")
    print(f"toy LOO (90 % rule, 10 sensors): modes {sorted(set(rs))}, POD RMS {np.mean(loo):.4f}, "
          f"mean-field RMS {np.mean(base):.4f}")
    print(f"1 mode, 3 sensors -> positions {np.round(x[clustered], 3)} m (clustering)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures")
    out = Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig_physics(out)
    fig_method(out)
    fig_physics(out, column=True)
    fig_method(out, column=True)


if __name__ == "__main__":
    main()
