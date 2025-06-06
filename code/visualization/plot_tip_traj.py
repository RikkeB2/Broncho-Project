#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def set_axes_equal(ax):
    """Make 3D axes have equal scale so spheres look like spheres."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d(x_middle - max_range/2, x_middle + max_range/2)
    ax.set_ylim3d(y_middle - max_range/2, y_middle + max_range/2)
    ax.set_zlim3d(z_middle - max_range/2, z_middle + max_range/2)

def main():
    parser = argparse.ArgumentParser(
        description="Plot a simulated tip trajectory from a .npy file (and optional centerline CSV)"
    )
    parser.add_argument('--sim', required=False, default="code/lib/engine/accumulated_points_2.npy",
                        help="Path to sim tip .npy (e.g. centerline_vs_fk_sim_tip.npy)")
    parser.add_argument('--centerline', required=False,
                        help="(Optional) Path to CSV with x_mm,y_mm,z_mm for overlay")
    args = parser.parse_args()

    # Load simulated tip
    tip_pts = np.load(args.sim)

    # Optionally load and plot the centerline
    raw_pts = None
    if args.centerline:
        import pandas as pd
        df = pd.read_csv(args.centerline)
        if set(('x_mm','y_mm','z_mm')).issubset(df.columns):
            raw_pts = df[['x_mm','y_mm','z_mm']].values
        else:
            print("Warning: centerline CSV missing x_mm,y_mm,z_mm columns; skipping overlay.")

    # Plot
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    if raw_pts is not None:
        ax.scatter(
            raw_pts[:,0], raw_pts[:,1], raw_pts[:,2],
            c='C0', marker='o', s=10, label='Centerline'
        )

    ax.scatter(
        tip_pts[:,0], tip_pts[:,1], tip_pts[:,2],
        c='C1', marker='x', s=20, label='Simulated Tip'
    )

    set_axes_equal(ax)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()