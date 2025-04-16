import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class PointCloudGenerator:
    def __init__(self, intrinsic_matrix, update_interval=50):
        self.intrinsic_matrix = intrinsic_matrix
        self.update_interval = update_interval
        self.update_count = 0
        self.pcd = o3d.geometry.PointCloud()
        self.reference_pcd = None  # Reference point cloud for alignment
        self.initial_transformation = None  # Store the first transformation matrix
    
    def process_depth_and_transform(self, depth_img, transformation_matrix):
        """Process depth image and apply transformation."""
        K = self.intrinsic_matrix
        height, width = depth_img.shape
        x, y = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
        x = (x - K[0, 2]) / K[0, 0]
        y = (y - K[1, 2]) / K[1, 1]
        
        # Apply a 180-degree rotation around the X-axis
        rotation_180_x = R.from_euler('x', 180, degrees=True).as_matrix()
        transformation_matrix[:3, :3] = transformation_matrix[:3, :3] @ rotation_180_x

        # Invert the Z translation (apply after rotation)
        transformation_matrix[2, 3] *= -1

        X = depth_img * x
        Y = depth_img * y
        Z = depth_img
        
        points_3d = np.vstack((X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())))
        points_world = (transformation_matrix @ points_3d).T[:, :3]
        
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points_world)
        
        # Save accumulated point cloud before alignment
        #self.save_accumulated_point_cloud(temp_pcd, filename="accumulated_point_cloud_before_alignment.npy")

        # Perform alignment using ICP
        if self.reference_pcd is not None:
            temp_pcd = self.apply_icp(self.reference_pcd, temp_pcd)
        else:
            print("Skipping alignment for this frame due to low fitness.")
    
        # Update the accumulated point cloud
        if not self.pcd.is_empty():
            self.pcd.points = o3d.utility.Vector3dVector(
                np.concatenate((np.asarray(self.pcd.points), np.asarray(temp_pcd.points)), axis=0)
            )
        else:
            self.pcd = temp_pcd

        # Debugging
        print(f"Accumulated point cloud size: {len(self.pcd.points)}")

        # Update the reference point cloud to the latest accumulated point cloud
        self.reference_pcd = self.pcd

        self.denoise_and_downsample()
        self.save_accumulated_point_cloud()

    def apply_icp(self, reference_pcd, target_pcd):
        """Align the target point cloud to the reference using ICP."""
        
        threshold = 0.089  # Distance threshold for ICP
        initial_transformation = np.identity(4)  # Replace with a better initial guess if available
        icp_result = o3d.pipelines.registration.registration_icp(
            target_pcd, reference_pcd, threshold,
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        print(f"ICP Fitness: {icp_result.fitness}, Inlier RMSE: {icp_result.inlier_rmse}")
        
        # Skip transformation if fitness is too low
        if icp_result.fitness < 0.3:  # Adjust the threshold as needed
            print("Low ICP fitness. Skipping transformation for this frame.")
            return target_pcd

        aligned_pcd = target_pcd.transform(icp_result.transformation)
        return aligned_pcd

    def denoise_and_downsample(self):
        """Apply denoising and downsampling to the point cloud."""
        if not self.pcd.is_empty():
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.008)
            cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            self.pcd = self.pcd.select_by_index(ind)
    
    def save_accumulated_point_cloud(self, pcd=None, filename="accumulated_point_cloud.npy"):
        """Save point cloud as a NumPy file in its own folder."""
        if pcd is None:
            pcd = self.pcd
        output_folder = os.path.join("pointclouds", "accumulated")
        os.makedirs(output_folder, exist_ok=True)
        file_path = os.path.join(output_folder, filename)
        np.save(file_path, np.asarray(pcd.points))
        print(f"Point cloud saved: {file_path}")
    
    def save_intermediate_pointcloud(self, depth_img, transformation_matrix, filename):
        """Generate a new point cloud from a depth image, apply ICP, and save it."""
        K = self.intrinsic_matrix
        height, width = depth_img.shape
        x, y = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
        x = (x - K[0, 2]) / K[0, 0]
        y = (y - K[1, 2]) / K[1, 1]
        
        X = depth_img * x
        Y = depth_img * y
        Z = depth_img
        
        points_3d = np.vstack((X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())))
        points_world = (transformation_matrix @ points_3d).T[:, :3]
        
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points_world)
        
        # Apply ICP if a reference point cloud exists
        if self.reference_pcd is not None:
            temp_pcd = self.apply_icp(self.reference_pcd, temp_pcd)
        else:
            print("Skipping ICP alignment due to missing reference point cloud.")
        
        # Save the aligned point cloud
        output_folder = "pointclouds/intermediate_pointclouds"
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, filename + ".npy"), np.asarray(temp_pcd.points))
        
        o3d.io.write_point_cloud(os.path.join(output_folder, filename + ".ply"), temp_pcd)
        print(f"Intermediate point cloud saved: {filename}.ply")