import numpy as np
import re
import matplotlib.pyplot as plt
import open3d as o3d  # For loading the centerline

# Load bronchoscope path data from file
with open("code/logs/kinematic.txt", "r") as f:
    content = f.read()

# Extract Pose matrices
pose_blocks = re.findall(r"Pose=\[\[(.*?)\]\], Alpha", content, re.DOTALL)

positions = []
for block in pose_blocks:
    try:
        rows = block.split("]\n [")
        matrix = np.array([[float(val) for val in row.strip("[] ").split()] for row in rows])
        translation = matrix[:3, 3]  # X, Y, Z translation
        positions.append(translation)
    except Exception as e:
        print("Skipping a malformed block:", e)
        continue

positions = np.array(positions)

# Load centerline data (assuming it's stored in a .ply file)
centerline_path = "code/generated_centerlines/centerlines_accumulated/final_accumulated_centerline.ply"
centerline_pcd = o3d.io.read_point_cloud(centerline_path)
centerline_points = np.asarray(centerline_pcd.points)

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot bronchoscope path
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linestyle='-', label="Bronchoscope Path")

# Plot centerline
ax.plot(centerline_points[:, 0], centerline_points[:, 1], centerline_points[:, 2], color='orange', linestyle='-', label="Centerline")

# Add labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Bronchoscope Path and Centerline")
ax.legend()

plt.tight_layout()
plt.show()