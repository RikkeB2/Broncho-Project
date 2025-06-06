import matplotlib.pyplot as plt
import numpy as np
import re
import open3d as o3d

# --- Load deviation data from log file ---
frame_indices = []
translation_devs = []
rotation_devs = []
deviation_positions = []

log_path = "code/logs/kinematic_deviation.txt"

with open(log_path) as f:
    lines = f.readlines()

for i in range(0, len(lines), 3):
    if i + 2 >= len(lines):
        break

    # Parse frame number
    match = re.search(r"Frame (\d+)", lines[i])
    if not match:
        continue
    frame = int(match.group(1))

    # Parse absolute translation deviation (vector x, y, z)
    trans_vals = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:e[+-]?\d+)?", lines[i + 1])
    if len(trans_vals) < 3:
        continue
    translation_vector = [float(trans_vals[0]), float(trans_vals[1]), float(trans_vals[2])]
    deviation_positions.append(translation_vector)

    trans_dev = np.linalg.norm(translation_vector)

    # Parse rotation deviation magnitude
    match = re.search(r"magnitude: ([0-9.eE+-]+)", lines[i + 2])
    if not match:
        continue
    rot_dev = float(match.group(1))

    frame_indices.append(frame)
    translation_devs.append(trans_dev)
    rotation_devs.append(rot_dev)

# --- Convert to per-frame deviations ---
translation_devs = np.diff(translation_devs, prepend=translation_devs[0])
rotation_devs = np.diff(rotation_devs, prepend=rotation_devs[0])

# --- Plot translation deviation per frame ---
plt.figure()
plt.plot(frame_indices, translation_devs, label="Translation Deviation Δ")
plt.xlabel("Frame")
plt.ylabel("Δ Translation Deviation (units)")
plt.title("Translation Drift Between Frames")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("translation_drift_per_frame.png")
plt.show()

# --- Plot rotation deviation per frame ---
plt.figure()
plt.plot(frame_indices, rotation_devs, label="Rotation Deviation Δ")
plt.xlabel("Frame")
plt.ylabel("Δ Rotation Deviation (degrees)")
plt.title("Rotation Drift Between Frames")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rotation_drift_per_frame.png")
plt.show()


