import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class PointCloudGenerator:
    def __init__(self, intrinsic_matrix, update_interval=50):
        self.intrinsic_matrix = intrinsic_matrix
        self.update_interval = update_interval
        self.update_count = 0  # Count how many times the point cloud has been updated
        self.pcd = o3d.geometry.PointCloud()  # The accumulated point cloud (without visualization)
        self.vis = None  # The Open3D visualization window

    def depth2pointcloud(self, depth_img2):
        """Convert a depth image to a 3D point cloud using correct scaling."""
        H, W = depth_img2.shape
        fx = self.intrinsic_matrix[0, 0] #Focal length in x direction
        fy = self.intrinsic_matrix[1, 1] #Focal length in y direction
        cx = self.intrinsic_matrix[0, 2] #Principal point in x direction
        cy = self.intrinsic_matrix[1, 2] #Principal point in y direction

        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        xv, yv = np.meshgrid(x, y)
        zv = depth_img2

        # Debugging: Print depth values
        print("Depth values (first 5):", zv.flatten()[:5])

        x = (xv - cx) * zv / fx
        y = (yv - cy) * zv / fy
        z = zv

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Debugging: Print first few points
        print("Generated points shape:", points.shape)
        print("First 5 valid points:", points[:5])  # Ensure points are being generated

        return points
    
    def voxel_filter(self, voxel_size=0.01):
        """Apply voxel grid filtering to the point cloud."""
        if len(self.pcd.points) == 0:
            return  # Nothing to filter

        print(f"Applying voxel filter with voxel size: {voxel_size}")
        filtered_pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        self.pcd = filtered_pcd
        print(f"Point cloud after voxel filter: {len(self.pcd.points)} points")

    def read_translation_log(self, log_file_path):
        """Read the translation log file and return the translations."""
        translations = []
        with open(log_file_path, "r") as file:
            for line in file:
                if line.startswith("Translation:"):
                    translation = np.fromstring(line[len("Translation: "):].strip()[1:-1], sep=' ')
                    translations.append(translation)
        return translations

    def apply_transformations(self, points, transformations):
        """Apply the transformations to the points."""
        transformed_points = points.copy()
        for transformation in transformations:
            transformed_points = (transformation @ transformed_points.T).T
        return transformed_points

    def update_point_cloud(self, depth_img2, log_file_path):
        """Update the stored point cloud and ensure it is properly formatted."""
        print(f"Depth Image Stats: Min={np.min(depth_img2)}, Max={np.max(depth_img2)}, Mean={np.mean(depth_img2)}")

        new_points = self.depth2pointcloud(depth_img2)
        print(f"Number of new points generated: {len(new_points)}")

        existing_points = np.asarray(self.pcd.points)

        # Debugging: Print existing points before update
        print(f"Existing points shape before update: {existing_points.shape}")

        # Read transformations from log file
        transformations = self.read_transformations_log(log_file_path)

        # Apply transformations to new points
        new_points = self.apply_transformations(new_points, transformations)

        # Append new points instead of overwriting
        combined_points = np.vstack((existing_points, new_points)) if existing_points.size else new_points

        # **Force correct data type for Open3D**
        combined_points = combined_points.astype(np.float32)

        self.pcd.points = o3d.utility.Vector3dVector(combined_points)

        # Apply voxel filter
        self.voxel_filter(voxel_size=0.002)

        # Debugging: Print combined points after update
        print(f"Combined points shape after update: {combined_points.shape}")
        print(f"Point cloud now contains {len(self.pcd.points)} points.")

        self.update_count += 1  # Track updates

    def save_pc(self, filename):
        """Save the accumulated point cloud to a file."""
        if len(self.pcd.points) == 0:
            print("Point cloud is empty! Nothing to save.")
            return
        
        print(f"Saving point cloud to {filename}...")

        # **Force Open3D to update its internal point cloud object**
        self.pcd.points = o3d.utility.Vector3dVector(np.asarray(self.pcd.points).astype(np.float32))

        # Debugging: Check if Open3D recognizes the points
        print(f"Final point count before saving: {len(self.pcd.points)}")

        try:
            o3d.io.write_point_cloud(filename, self.pcd)
            print("Point cloud saved successfully.")
        except Exception as e:
            print(f"Error saving point cloud: {e}")

        # Save as NumPy array for debugging
        np.save(filename + ".npy", np.asarray(self.pcd.points))
        print(f"Point cloud also saved as NumPy array to {filename}.npy")

    def save_copy(self, filename):
        """Save a copy of the point cloud for debugging."""
        if len(self.pcd.points) == 0:
            print("Point cloud is empty! Nothing to save.")
            return
        
       # print(f"Saving debug point cloud to {filename}...")
        #o3d.io.write_point_cloud(filename, self.pcd)
        #print("Debug point cloud saved.")

    def show(self):
        """Open Open3D visualization and block until closed."""
        if self.vis is None:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window()
            self.vis.add_geometry(self.pcd)

            # Register key callbacks
            self.vis.register_key_callback(256, self.close_visualizer)  # ESC key
            self.vis.register_key_callback(99, self.close_visualizer)   # 'C' key

            print("Opened Open3D visualization window. Press ESC or C to close.")

        # **Block execution until the user closes the window**
        self.vis.run()  
        self.close_visualizer()  # Ensure proper cleanup
   
    def close_visualizer(self, vis=None):
        """Close the Open3D visualization window and restore OpenGL context."""
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
            print("Closed Open3D visualization window.")

    def read_transformations_log(self, log_file_path):
        """Read the translation log file and return the transformations as rotation matrices."""
        transformations = []
        with open(log_file_path, "r") as file:
            for line in file:
                if line.startswith("Translation:"):
                    translation = np.fromstring(line[len("Translation: "):].strip()[1:-1], sep=' ')
                    # Assuming the translation values are roll, pitch, yaw
                    rotation_matrix = R.from_euler('xyz', translation).as_matrix()
                    transformations.append(rotation_matrix)
        return transformations

    def generate_point_cloud_from_transformations(self, transformations):
        """Generate a point cloud from the transformations."""
        points = []
        for transformation in transformations:
            # Generate a point at the origin and apply the transformation
            point = np.array([0, 0, 0])
            transformed_point = transformation @ point
            points.append(transformed_point)
        return np.array(points)

    def create_point_cloud_from_log(self, log_file_path):
        """Create a point cloud from the translation log."""
        transformations = self.read_transformations_log(log_file_path)
        points = self.generate_point_cloud_from_transformations(transformations)
        self.pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        print(f"Generated point cloud with {len(points)} points from log.")