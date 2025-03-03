import open3d as o3d
import numpy as np

class PointCloudGenerator:
    def __init__(self, intrinsic_matrix, update_interval=10):
        self.intrinsic_matrix = intrinsic_matrix
        self.update_interval = update_interval
        self.update_count = 0  # Count how many times the point cloud has been updated
        self.pcd = o3d.geometry.PointCloud()  # The accumulated point cloud (without visualization)

    def depth2pointcloud(self, depth_img2):
        """Convert a depth image to a 3D point cloud using correct scaling."""
        H, W = depth_img2.shape
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]

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


    def update_point_cloud(self, depth_img2):
        """Update the stored point cloud in the background (without visualization)."""
        print(f"Depth Image Stats: Min={np.min(depth_img2)}, Max={np.max(depth_img2)}, Mean={np.mean(depth_img2)}")

        new_points = self.depth2pointcloud(depth_img2)
        print(f"Number of new points generated: {len(new_points)}")  # Add this line

        existing_points = np.asarray(self.pcd.points)

        # Debugging: Print existing points before update
        print(f"Existing points shape before update: {existing_points.shape}")

        # Append new points instead of overwriting
        combined_points = np.vstack((existing_points, new_points)) if existing_points.size else new_points
        self.pcd.points = o3d.utility.Vector3dVector(combined_points)

        # Debugging: Print combined points after update
        print(f"Combined points shape after update: {combined_points.shape}")

        self.update_count += 1  # Track updates
        print(f"Updated point cloud {self.update_count} times.")
        print(f"Point cloud updated. Current total points: {len(self.pcd.points)}")

        # Save a copy of the point cloud after each update for debugging
        self.save_copy(f"debug_point_cloud_{self.update_count}.pcd")

    def save_pc(self, filename):
        """Save the accumulated point cloud to a file."""
        if len(self.pcd.points) == 0:
            print("Point cloud is empty! Nothing to save.")
            return
        
        print(f"Saving point cloud to {filename}...")
        o3d.io.write_point_cloud(filename, self.pcd)
        print("Point cloud saved.")

    def save_copy(self, filename):
        """Save a copy of the point cloud for debugging."""
        if len(self.pcd.points) == 0:
            print("Point cloud is empty! Nothing to save.")
            return
        
        print(f"Saving debug point cloud to {filename}...")
        o3d.io.write_point_cloud(filename, self.pcd)
        print("Debug point cloud saved.")

    def show(self):
        """Visualize the point cloud only when called."""
        if len(self.pcd.points) == 0:
            print("Point cloud is empty! No visualization.")
            return
        
        print("Displaying the point cloud...")
        o3d.visualization.draw_geometries([self.pcd])  # Show the accumulated point cloud