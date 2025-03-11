import numpy as np
import math
import sys
import os
from PIL import Image
import open3d as o3d

def euler_to_rotation_matrix(roll, pitch, yaw):
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    rotation_x = np.array([[1, 0, 0],
                           [0, cos_r, -sin_r],
                           [0, sin_r, cos_r]])

    rotation_y = np.array([[cos_p, 0, sin_p],
                           [0, 1, 0],
                           [-sin_p, 0, cos_p]])

    rotation_z = np.array([[cos_y, -sin_y, 0],
                           [sin_y, cos_y, 0],
                           [0, 0, 1]])

    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    return rotation_matrix

def back_project(pixel_coord, depth, camera_intrinsics):
    """Back-projects a pixel coordinate to 3D given depth and intrinsics."""
    x, y = pixel_coord
    fx, fy, cx, cy = camera_intrinsics  # Unpack intrinsics

    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z])

def get_depth_for_pixel(depth_array, index):
    """Placeholder: Implement based on your depth image and rotation log."""
    height, width = depth_array.shape
    # Assuming that the index corresponds to a pixel in row-major order.
    # This is a very simple and potentially incorrect assumption,
    # and needs to be replaced with a proper synchronization logic.
    y = index // width
    x = index % width

    if 0 <= y < height and 0 <= x < width:
        return depth_array[y, x]
    else:
        return 0  # Or some other invalid depth value

def process_camera_logs(rotation_log_file, translation_log_file, depth_image_path, camera_intrinsics):
    """
    Processes camera logs, calculates transforms, 
    retrieves depth from the image, and back-projects to 3D.
    """
    with open(rotation_log_file, 'r') as rot_f, open(translation_log_file, 'r') as trans_f:
        rotation_lines = rot_f.readlines()
        translation_lines = trans_f.readlines()

    translation_data = []
    for line in translation_lines:
        translation_str = line.split('[')[1].split(']')[0]
        translation_values = list(map(float, translation_str.split()))
        translation_data.append(np.array(translation_values))

    translation_index = 0

    # Load depth image
    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image)

    # Point Cloud Transformation and Back-projection
    point_cloud_3d = []  # To store the 3D point cloud

    for i, rotation_line in enumerate(rotation_lines):
        # Skip non-numeric lines
        if not rotation_line.strip() or not rotation_line[0].isdigit():
            continue

        timestep, roll, pitch, yaw = map(float, rotation_line.strip().split(','))
        rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)

        if i % 100 == 0 and translation_index < len(translation_data):
            translation_vector = translation_data[translation_index]
            translation_index += 1
        else:
            if translation_index > 0:
                translation_vector = translation_data[translation_index - 1]
            else:
                translation_vector = np.array([0.0, 0.0, 0.0])

        # Assuming you have a function to get depth for a pixel
        depth = get_depth_for_pixel(depth_array, i)

        # Calculate projected point for back projection.
        fov_radians = 2 * np.arctan(100 / 181.9375)
        focal_length_y = 100 / np.tan(fov_radians / 2)
        focal_length_x = focal_length_y
        cx = 200 / 2
        cy = 200 / 2
        projected_point = np.array([(i % 200), (i // 200)]) #example projection. replace with your actual projection.

        # Back-project and transform to world coordinates
        point_3d_camera = back_project(projected_point, depth, camera_intrinsics)  # Back-project
        point_3d_world = np.dot(rotation_matrix, point_3d_camera) + translation_vector  # Transform
        point_cloud_3d.append(point_3d_world)

    return np.array(point_cloud_3d)

# File paths
rotation_log_file = os.path.join(os.path.dirname(__file__), '..', 'code', 'camera_orientation_log.txt')
translation_log_file = os.path.join(os.path.dirname(__file__), '..', 'code', 'translation_log.txt')
depth_image_path = "depth.png"

# Camera intrinsics
fov_radians = 2 * np.arctan(100 / 181.9375)
focal_length_y = 100 / np.tan(fov_radians / 2)
focal_length_x = focal_length_y
cx = 200 / 2
cy = 200 / 2
camera_intrinsics = (focal_length_x, focal_length_y, cx, cy)

# Ensure files exist
if not os.path.exists(rotation_log_file):
    print(f"Error: {rotation_log_file} does not exist.")
    exit()

if not os.path.exists(translation_log_file):
    print(f"Error: {translation_log_file} does not exist.")
    exit()

if not os.path.exists(depth_image_path):
    print(f"Error: {depth_image_path} does not exist.")
    exit()

# Process logs and create point cloud
point_cloud_3d = process_camera_logs(rotation_log_file, translation_log_file, depth_image_path, camera_intrinsics)

# Visualize the point cloud (using Open3D)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud_3d)
o3d.visualization.draw_geometries([pcd])