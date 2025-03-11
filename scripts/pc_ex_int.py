import numpy as np
import open3d as o3d
import cv2
import sys
import os

# Add the path to the camera module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code', 'lib', 'engine'))

from camera import fixedCamera  # Import the fixedCamera class

def depth_map_to_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix):
    """
    Converts a depth map to a point cloud.

    Args:
        depth_map: A NumPy array representing the depth map.
        intrinsic_matrix: The 3x3 intrinsic matrix.
        extrinsic_matrix: The 4x4 extrinsic matrix.

    Returns:
        A NumPy array representing the point cloud (Nx3).
    """

    height, width = depth_map.shape
    points = []

    # Inverse Intrinsic Matrix
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)

    # Inverse Extrinsic Matrix
    extrinsic_inv = np.linalg.inv(extrinsic_matrix)

    for v in range(height):
        for u in range(width):
            depth = depth_map[v, u]
            if depth > 0:  # Valid depth
                # Unproject to Camera Coordinates
                pixel_coords = np.array([[u, v, 1]]).T
                camera_coords = depth * np.dot(intrinsic_inv, pixel_coords)
                camera_coords_homogeneous = np.vstack((camera_coords, 1))

                # Transform to World Coordinates
                world_coords_homogeneous = np.dot(extrinsic_inv, camera_coords_homogeneous)
                world_coords = world_coords_homogeneous[:3].flatten()

                points.append(world_coords)

    return np.array(points)

# Intrinsics
fov_deg = 2 * np.arctan(100 / 181.9375) / np.pi * 180
fov_rad = np.radians(fov_deg)
aspect = 1.0  # Assuming square pixels
near_val = 0.00001
far_val = 100

# Load the depth image
depth_map = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if depth_map is None:
    print("Error: Could not load depth image 'depth.png'")
    exit()

image_height, image_width = depth_map.shape

focal_length = (image_height / 2) / np.tan(fov_rad / 2) # calculate the focal length

cx = image_width / 2
cy = image_height / 2

intrinsic_matrix = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
])

# Create an instance of fixedCamera
physics_server = p.connect(p.DIRECT)  # or p.GUI for graphical interface
camera = fixedCamera(dis=1.0, physics_server=physics_server)

# Get the extrinsic matrix from the camera
extrinsic_matrix = camera.getExtrinsic()

# Generate Point Cloud
point_cloud = depth_map_to_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix)

# Correctly create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Colorize point cloud based on Z-values
z_values = point_cloud[:, 2]  # Extract Z-coordinates
min_z, max_z = np.min(z_values), np.max(z_values)

colors = np.zeros((len(point_cloud), 3))  # Create an array of zeros (default black)

if max_z > min_z:  # Avoid division by zero if all Z values are the same
    normalized_z = (z_values - min_z) / (max_z - min_z)  # Normalize Z values to [0, 1]
    colors[:, 0] = normalized_z  # Map Z to red channel

# Assign colors to the point cloud
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])