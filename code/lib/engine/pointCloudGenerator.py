import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class PointCloudGenerator:
    def __init__(self, intrinsic_matrix, update_interval=50):
        self.intrinsic_matrix = intrinsic_matrix
        self.update_interval = update_interval
        self.update_count = 0 # Count how many times the point cloud has been updated
        self.pcd = o3d.geometry.PointCloud() # The accumulated point cloud (without visualization)
        self.vis = None # The Open3D visualization window
        self.previous_pcd = None # The previous point cloud for ICP
        self.R = np.eye(3)  # Initialize rotation matrix
        self.T = np.zeros(3) # Initialize translation vector
        self.intermediate_point_clouds = []  # Store intermediate point clouds for combining

    def get_transformation_matrix(self, R, T):
        """Get a transformation matrix from a rotation matrix and translation vector."""
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = T
        return transformation

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
    
    def save_intermediate_pc(self, depth_img2, filename):
        """Generate a new point cloud from a depth image and save it to a file."""
        
        # Generate a new point cloud from the depth image
        points = self.depth2pointcloud(depth_img2)

        # Convert points to homogeneous coordinates
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

        transformation_matrix = self.get_transformation_matrix(self.R, self.T)
        transformed_points = (transformation_matrix @ homogeneous_points.T).T[:, :3]
        
        # Create a new point cloud object
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(transformed_points.astype(np.float32))
        
        # Save the new point cloud to a file
        o3d.io.write_point_cloud(filename, new_pcd)
        print("New point cloud saved successfully.")
        
        # Save as NumPy array for debugging
        np.save(filename + ".npy", np.asarray(new_pcd.points))
        print(f"New point cloud also saved as NumPy array to {filename}.npy")

        # Append the new point cloud to the list of intermediate point clouds
        self.intermediate_point_clouds.append(new_pcd)

    def combine_point_clouds(self):
        """Combine all intermediate point clouds into the main point cloud."""
        combined_points = []
        for pcd in self.intermediate_point_clouds:
            combined_points.extend(np.asarray(pcd.points))

        if combined_points:
            combined_points_array = np.array(combined_points).astype(np.float32)  # Convert to NumPy array
            self.pcd.points = o3d.utility.Vector3dVector(combined_points_array)
            print(f"Combined {len(self.intermediate_point_clouds)} point clouds into the main point cloud.")
            self.intermediate_point_clouds = []  # Clear the list after combining

            # Save the combined point cloud as a NumPy array
            np.save("combined_point_cloud.npy", combined_points_array)
            print("Combined point cloud saved as combined_point_cloud.npy")
        else:
            print("No intermediate point clouds to combine.")




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
        new_points = self.depth2pointcloud(depth_img2)
        existing_points = np.asarray(self.pcd.points)

        print(f"Existing points shape before update: {existing_points.shape}")

        # Apply the transformations from the log file
        transformations = self.read_transformations_log(log_file_path)
        
        # Apply the transformations to the new points
        new_points = self.apply_transformations(new_points, transformations)

        # Combine the existing and new points
        # Append new points instead of overwriting
        combined_points = np.vstack((existing_points, new_points)) if existing_points.size else new_points
        
        # **Force correct data type for Open3D**
        combined_points = combined_points.astype(np.float32)

        self.pcd.points = o3d.utility.Vector3dVector(combined_points)

        # ICP Implementation
        if self.previous_pcd is not None and len(self.pcd.points) > 0 and len(self.previous_pcd.points) > 0:
            threshold = 0.02
            max_iteration = 200
            reg_p2p = o3d.pipelines.registration.registration_icp(
                self.previous_pcd, self.pcd, threshold, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
            )

            print("ICP Fitness:", reg_p2p.fitness)
            print("ICP Inlier RMSE:", reg_p2p.inlier_rmse)

            self.pcd.transform(reg_p2p.transformation)
            print("ICP applied.")

        # Voxel filter after ICP
        self.voxel_filter(voxel_size=0.002)

        # Debugging: Print combined points after update
        print(f"Combined points shape after update: {combined_points.shape}")
        print(f"Point cloud now contains {len(self.pcd.points)} points.")

        self.previous_pcd = o3d.geometry.PointCloud(self.pcd)
        self.update_count += 1

    def save_pc(self, filename):
        """Save the accumulated point cloud to a file."""

        if len(self.pcd.points) == 0:
            print("Point cloud is empty! Nothing to save.")
            return

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

    def show(self):
        """Open Open3D visualization and block until closed."""
        if self.vis is None:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window()
            self.vis.add_geometry(self.pcd)

            self.vis.register_key_callback(256, self.close_visualizer) # ESC key
            self.vis.register_key_callback(99, self.close_visualizer) # C key

            print("Opened Open3D visualization window. Press ESC or C to close.")

        # **Block execution until the user closes the window**
        self.vis.run()
        self.close_visualizer()

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
                    rotation_matrix = R.from_euler('xyz', translation).as_matrix()
                    transformations.append(rotation_matrix)
        return transformations
    
