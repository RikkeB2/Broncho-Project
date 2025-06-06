import numpy as np
import open3d as o3d
import os
import glob

def load_pointcloud(file_path, color=None):
    """
    Loads a point cloud from a file and optionally assigns a color.

    Parameters:
    - file_path: Path to the point cloud file (.ply or .npy).
    - color: Optional RGB color to assign to the point cloud.

    Returns:
    - pcd: The loaded Open3D PointCloud object.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        # Handle .ply files
        if file_path.endswith(".ply"):
            pcd = o3d.io.read_point_cloud(file_path)

            if pcd.is_empty():
                print(f"Error: The point cloud file {file_path} is empty.")
                return None

        # Handle .npy files
        elif file_path.endswith(".npy"):
            points = np.load(file_path)

            if points.size == 0:
                print(f"Error: The .npy file {file_path} is empty.")
                return None
            # Invert Z axis
            points = points.copy()
            points[:, 2] *= -1
            # Convert numpy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

        else:
            print(f"Error: Unsupported file format for {file_path}")
            return None

        # Assign color if provided
        if color:
            pcd.paint_uniform_color(color)

        return pcd

    except Exception as e:
        print(f"Error: Could not read point cloud {file_path}: {e}")
        return None

def show_multiple_pointclouds(file_paths_or_folder):
    """
    Visualizes multiple point clouds together. Can handle individual files or a folder containing point clouds.

    Parameters:
    - file_paths_or_folder: List of paths to the point cloud files or a folder containing point clouds.
    """
    geometries = []

    # If input is a folder, gather all .ply and .npy files
    if isinstance(file_paths_or_folder, str) and os.path.isdir(file_paths_or_folder):
        file_paths = glob.glob(os.path.join(file_paths_or_folder, "*.ply")) + glob.glob(os.path.join(file_paths_or_folder, "*.npy"))
    else:
        file_paths = file_paths_or_folder

    for file_path in file_paths:
        # Load each point cloud
        color = [np.random.rand(), np.random.rand(), np.random.rand()]  # Assign a random color to each point cloud
        pcd = load_pointcloud(file_path, color=color)
        if pcd:
            geometries.append(pcd)

    if geometries:
        # Visualize all point clouds together
        o3d.visualization.draw_geometries(geometries, window_name="Point Cloud Viewer", width=1200, height=1200,
                                          point_show_normal=False)
    else:
        print("Error: No valid point clouds to display.")

if __name__ == "__main__":

    file_paths = [
        "code/pointclouds/accumulated/accumulated_point_cloud_frame_200.npy"
    ]
    #show_multiple_pointclouds(file_paths)

    folder_path = "code/pointclouds/intermediate_pointclouds"
    show_multiple_pointclouds(folder_path)