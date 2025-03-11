import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Camera log data (replace with your actual data)
camera_log = [
    {"Timestep": 0, "Translation": [0, 0, 0], "Rotation": [[1.0, -0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 100, "Translation": [1, 1, 1], "Rotation": [[0.5403023058681398, -0.8414709848078965, 0.0], [0.8414709848078965, 0.5403023058681398, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 200, "Translation": [2, 2, 2], "Rotation": [[-0.4161468365471424, -0.9092974268256817, 0.0], [0.9092974268256817, -0.4161468365471424, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 300, "Translation": [3, 3, 3], "Rotation": [[-0.9899924966004454, -0.1411200080598672, 0.0], [0.1411200080598672, -0.9899924966004454, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 400, "Translation": [4, 4, 4], "Rotation": [[-0.6536436208636119, 0.7568024953079282, 0.0], [-0.7568024953079282, -0.6536436208636119, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 500, "Translation": [5, 5, 5], "Rotation": [[0.28366218546322625, 0.9589242746631385, 0.0], [-0.9589242746631385, 0.28366218546322625, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 600, "Translation": [6, 6, 6], "Rotation": [[0.960170286650366, 0.27941549819892586, 0.0], [-0.27941549819892586, 0.960170286650366, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 700, "Translation": [7, 7, 7], "Rotation": [[0.7539022543433046, -0.6569865987187891, 0.0], [0.6569865987187891, 0.7539022543433046, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 800, "Translation": [8, 8, 8], "Rotation": [[-0.14550003380861354, -0.9893582466233818, 0.0], [0.9893582466233818, -0.14550003380861354, 0.0], [0.0, 0.0, 1.0]]},
    {"Timestep": 900, "Translation": [9, 9, 9], "Rotation": [[-0.9111302618846769, -0.4121184852417566, 0.0], [0.4121184852417566, -0.9111302618846769, 0.0], [0.0, 0.0, 1.0]]}
]

# Extract data for plotting
timesteps = [entry["Timestep"] for entry in camera_log]
x_translations = [entry["Translation"][0] for entry in camera_log]
y_translations = [entry["Translation"][1] for entry in camera_log]
z_translations = [entry["Translation"][2] for entry in camera_log]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the translation data
ax.plot(x_translations, y_translations, z_translations)

# Set axis labels
ax.set_xlabel('X Translation')
ax.set_ylabel('Y Translation')
ax.set_zlabel('Z Translation')

# Set plot title
ax.set_title('Camera Translation Over Time')

# Show the plot
plt.show()

def read_logs(orientation_log_path, translation_rotation_log_path):
    orientations = []
    translations = []

    with open(orientation_log_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            timestep = int(parts[0])
            roll = float(parts[1])
            pitch = float(parts[2])
            yaw = float(parts[3])
            orientations.append((timestep, roll, pitch, yaw))

    with open(translation_rotation_log_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            timestep = int(parts[0])
            translation = np.array([float(x.strip()) for x in parts[1].strip('[]').split(',')])
            rotation = np.array([float(x.strip()) for x in parts[2].strip('[]').replace('], [', ',').split(',')]).reshape((3, 3))
            translations.append((timestep, translation, rotation))

    return orientations, translations

def generate_point_clouds(orientations, translations):
    point_clouds = []

    for timestep, translation, rotation in translations:
        # Create a point cloud
        pc = o3d.geometry.PointCloud()

        # Generate points (example: a simple cube)
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])

        # Create a 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation

        # Apply the transformation
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = np.dot(transformation, points_homogeneous.T).T[:, :3]

        # Set points to the point cloud
        pc.points = o3d.utility.Vector3dVector(transformed_points)
        point_clouds.append(pc)

    return point_clouds

def main():
    orientation_log_path = "camera_orientation_log.txt"
    translation_rotation_log_path = "translation_rotation_log.txt"

    orientations, translations = read_logs(orientation_log_path, translation_rotation_log_path)
    point_clouds = generate_point_clouds(orientations, translations)

    # Visualize point clouds
    o3d.visualization.draw_geometries(point_clouds)

if __name__ == "__main__":
    main()