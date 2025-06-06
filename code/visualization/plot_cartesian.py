import numpy as np
import matplotlib.pyplot as plt
import re

positions = []
with open("code/logs/cartesian_data.txt") as f:
    for line in f:
        match = re.search(r"Position: \[([^\]]+)\]", line)
        if match:
            pos = np.fromstring(match.group(1), sep=' ')
            positions.append(pos)
positions = np.array(positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:,0], positions[:,1], positions[:,2], marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Trajectory")
plt.show()

plt.plot(positions[:,0], label='X')
plt.plot(positions[:,1], label='Y')
plt.plot(positions[:,2], label='Z')
plt.xlabel('Frame')
plt.ylabel('Position')
plt.legend()
plt.title("Position Components Over Time")
plt.show()

euclid_dist = np.linalg.norm(positions - positions[0], axis=1)
plt.plot(euclid_dist)
plt.xlabel('Frame')
plt.ylabel('Distance from Start [units]')
plt.title("Euclidean Distance from Start Over Time")
plt.show()