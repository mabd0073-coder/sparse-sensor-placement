import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- COMPATIBILITY PATCH FOR NEWER NUMPY VERSIONS ---
legacy_aliases = [
    ('long', int), ('ulong', int), ('longlong', int), ('ulonglong', int),
    ('uintc', int), ('intc', int), ('int', int), ('float', float),
    ('complex', complex), ('object', object)
]
for alias, target in legacy_aliases:
    if not hasattr(np, alias):
        setattr(np, alias, target)
# ----------------------------------------------------

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("Note: scipy not found, falling back to slower brute-force distance checks.")

# Start timer
start = time.time()

# Distance threshold for flagging (in mm — assumes coordinates are in mm)
THRESHOLD_MM = 30.0

# Target spacing between resampled points across each triangle face (in mm)
DENSIFY_SPACING_MM = 20

# Files whose points get checked against reference files
TARGET_FILES = ["SD1.txt", "SD2.txt", "SD3.txt"]
REFERENCE_FILES = ["SD1A.txt", "SD1B.txt", "SD1C.txt", "SD1D.txt", "SD1E.txt", "SD1F.txt"]


def parse_vertices(filepath):
    """Extract (x, y, z) vertex coordinates from an ASCII STL-style txt file."""
    points = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append((x, y, z))
                except (IndexError, ValueError):
                    # Skip malformed vertex lines instead of crashing
                    continue
    return np.array(points)


def parse_triangles(filepath):
    """Extract triangles (Nx3x3: N triangles, 3 vertices, xyz) from an ASCII STL-style txt file."""
    points = parse_vertices(filepath)
    n_complete = (len(points) // 3) * 3
    if n_complete != len(points):
        print(f"Warning: {filepath} has a vertex count not divisible by 3 "
              f"({len(points)} vertices) — trailing incomplete triangle discarded.")
    return points[:n_complete].reshape(-1, 3, 3)


def densify_triangles(triangles, spacing):
    """Resample each triangle with a barycentric grid of points spaced roughly `spacing` apart."""
    all_points = []
    for tri in triangles:
        p0, p1, p2 = tri
        edge_lengths = (
            np.linalg.norm(p1 - p0),
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p0 - p2),
        )
        max_edge = max(edge_lengths)
        n = max(1, int(np.ceil(max_edge / spacing)))

        for i in range(n + 1):
            for j in range(n + 1 - i):
                u = i / n
                v = j / n
                w = 1.0 - u - v
                all_points.append(w * p0 + u * p1 + v * p2)

    if not all_points:
        return np.empty((0, 3))
    return np.array(all_points)


def build_reference_cloud():
    """Combine densified points from reference files into one reference point cloud."""
    ref_point_list = []
    ref_points_by_file = {}
    for ref_name in REFERENCE_FILES:
        if os.path.exists(ref_name):
            triangles = parse_triangles(ref_name)
            ref_pts = densify_triangles(triangles, DENSIFY_SPACING_MM)
            ref_point_list.append(ref_pts)
            ref_points_by_file[ref_name] = ref_pts
        else:
            print(f"Warning: reference file {ref_name} not found — skipping it.")

    if ref_point_list:
        combined = np.vstack(ref_point_list)
    else:
        combined = np.empty((0, 3))
    return combined, ref_points_by_file


def flag_points(points, ref_points, tree):
    """Return a boolean array: True if the point is within THRESHOLD_MM of any reference point."""
    if len(ref_points) == 0:
        return np.zeros(len(points), dtype=bool)

    if HAVE_SCIPY:
        dist, _ = tree.query(points, k=1)
        return dist <= THRESHOLD_MM
    else:
        flags = np.zeros(len(points), dtype=bool)
        for i, p in enumerate(points):
            d = np.sqrt(((ref_points - p) ** 2).sum(axis=1)).min()
            flags[i] = d <= THRESHOLD_MM
        return flags


def plot_points(points, flags, title, output_path):
    """Save a 3D scatter plot of points (red = flagged, blue = not) without displaying it."""
    colors = np.where(flags, "red", "blue")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=10, alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_points_single_color(points, title, output_path, color="green"):
    """Save a 3D scatter plot of points in a single color without displaying it."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=10, alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    ref_points, ref_points_by_file = build_reference_cloud()
    print(f"Reference cloud built from reference files: {len(ref_points)} points.")

    for ref_name, ref_pts in ref_points_by_file.items():
        if len(ref_pts) > 0:
            plot_name = ref_name.split(".")[0] + "_plot.png"
            plot_points_single_color(ref_pts, f"{ref_name} (reference)", plot_name, color="green")

    tree = None
    if HAVE_SCIPY and len(ref_points) > 0:
        tree = cKDTree(ref_points)

    all_points = []
    all_flags = []

    for x in os.listdir():
        if x.endswith(".txt"):
            print("Converting " + x + " to .csv file.")

            if x in TARGET_FILES:
                triangles = parse_triangles(x)
                points = densify_triangles(triangles, DENSIFY_SPACING_MM)
            else:
                points = parse_vertices(x)

            if x in TARGET_FILES and len(points) > 0:
                flags = flag_points(points, ref_points, tree)
                all_points.append(points)
                all_flags.append(flags)

                plot_name = x.split(".")[0] + "_plot.png"
                plot_points(points, flags, f"{x}: within {THRESHOLD_MM}mm of reference (red) vs not (blue)", plot_name)
            else:
                flags = np.array([""] * len(points))  # No flag computed for non-target files

            csv_ = x.split(".")
            new_csv_name = csv_[0] + ".csv"

            with open(new_csv_name, "w") as f:
                f.write("x,y,z,near_reference\n")
                for p, flag in zip(points, flags):
                    f.write(f"{p[0]},{p[1]},{p[2]},{flag}\n")

    # Plot all target points: red = flagged (within threshold), blue = not flagged
    if all_points:
        points_combined = np.vstack(all_points)
        flags_combined = np.concatenate(all_flags)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        colors = np.where(flags_combined, "red", "blue")
        ax.scatter(
            points_combined[:, 0],
            points_combined[:, 1],
            points_combined[:, 2],
            c=colors,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Points within {THRESHOLD_MM}mm of reference (red) vs not (blue)")
        fig.tight_layout()
        fig.savefig("flagged_points_plot.png", dpi=200)
        print("Saved plot to flagged_points_plot.png")
        plt.show()

        pd.DataFrame({
            "x": points_combined[:, 0], 
            "y": points_combined[:, 1], 
            "z": points_combined[:, 2],
            "excluded": flags_combined
        }).to_csv("all_points_with_excluded.csv", index=False)

        print("Saved all points to all_points.csv")
    else:
        print("No target points found to plot (SD1/SD2/SD3 not found or empty).")

    elapsed = "{:.2f}".format(time.time() - start)
    print("")
    print("Process complete. Run time: " + elapsed + " seconds.")


if __name__ == "__main__":
    main()