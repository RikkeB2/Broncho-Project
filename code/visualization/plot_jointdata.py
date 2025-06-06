import re
import matplotlib.pyplot as plt

# File paths
forward_file = "code\logs\joint_data.txt"

def read_joint_file(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            # Find numbers inside brackets
            match = re.search(r"\[([^\]]+)\]", line)
            if match:
                nums = [float(n) for n in match.group(1).split(",")]
                if len(nums) == 3:
                    data.append(nums)
    if not data:
        print(f"No data found in {file_path}")
        return [[], [], []]
    return list(zip(*data))  # Transpose to get lists for each joint

forward = read_joint_file(forward_file)
# Multiply translation (index 0) by 1000
forward = list(forward)
forward[0] = [v * 1 for v in forward[0]]

joint_names = ["Translation", "Bending", "Rotation"]
joint_units = ["[cm]", "[rad]", "[rad]"]

plt.figure(figsize=(8, 10))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(forward[i], label=f"Forward {joint_names[i]}")
    plt.title(f"Forward {joint_names[i]}")
    plt.xlabel("Step")
    plt.ylabel(f"{joint_names[i]} {joint_units[i]}")
    plt.legend()

plt.tight_layout(rect=[0, 0, 0.97, 1])  # Leave a bit more space on the right
plt.show()