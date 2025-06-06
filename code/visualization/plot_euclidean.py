import re
import matplotlib.pyplot as plt
import numpy as np

frames = []
mean_distances = []
max_distances = []

with open("logs/centerline_euclidean_distance.txt") as f:
    for line in f:
        mean_match = re.search(r"Frame (\d+) - Euclidean mean distance: ([\d\.]+)", line)
        max_match = re.search(r"Frame (\d+) - Euclidean max distance: ([\d\.]+)", line)
        if mean_match:
            frames.append(int(mean_match.group(1)))
            mean_distances.append(float(mean_match.group(2)))
        if max_match:
            max_distances.append(float(max_match.group(2)))

# Plot mean distances
plt.plot(frames, mean_distances, marker='o', label='Mean')
plt.plot(frames, max_distances, marker='x', label='Max')
plt.xlabel("Frame")
plt.ylabel("Euclidean Distance [m]")
plt.title("Mean and Max Euclidean Distance to Centerline Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print absolute mean and max
print(f"Absolute mean of mean distances: {np.mean(mean_distances):.6f} m")
print(f"Absolute max of max distances: {np.max(max_distances):.6f} m")