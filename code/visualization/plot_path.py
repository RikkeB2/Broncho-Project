import numpy as np
import re
import matplotlib.pyplot as plt

# Load data from file
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

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linestyle='-')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Bronchoscope 3D Path")
plt.tight_layout()
plt.show()