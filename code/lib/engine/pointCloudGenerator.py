import open3d as o3d
import numpy as np

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
        """Update the stored point cloud and ensure it is properly formatted."""
        print(f"Depth Image Stats: Min={np.min(depth_img2)}, Max={np.max(depth_img2)}, Mean={np.mean(depth_img2)}")

        new_points = self.depth2pointcloud(depth_img2)
        print(f"Number of new points generated: {len(new_points)}")

        existing_points = np.asarray(self.pcd.points)

        # Debugging: Print existing points before update
        print(f"Existing points shape before update: {existing_points.shape}")

        # Append new points instead of overwriting
        combined_points = np.vstack((existing_points, new_points)) if existing_points.size else new_points

        # **Force correct data type for Open3D**
        combined_points = combined_points.astype(np.float32)

        self.pcd.points = o3d.utility.Vector3dVector(combined_points)

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

            # **RESTORE OpenGL context for pyrender**
            gl.glFlush()  # Ensures OpenGL commands finish before pyrender resumes
            print("OpenGL context restored.")