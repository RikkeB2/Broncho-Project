import open3d as o3d
import time
import numpy as np

class PointCloudGenerator:
    def __init__(self, intrinsic_matrix):
        # Initialize the point cloud generator with the camera's intrinsic matrix
        self.intrinsic_matrix = intrinsic_matrix
        self.vis = o3d.visualization.Visualizer()  # Create a visualizer object
        self.vis.create_window()  # Create a window for visualization
        self.pcd = o3d.geometry.PointCloud()  # Create an empty point cloud object
        self.vis.add_geometry(self.pcd)  # Add the point cloud to the visualizer

    def depth2pointcloud(self, depth_img):
        # Convert a depth image to a point cloud
        H, W = depth_img.shape  # Get the height and width of the depth image
        fx = self.intrinsic_matrix[0, 0]  # Focal length in x direction
        fy = self.intrinsic_matrix[1, 1]  # Focal length in y direction
        cx = self.intrinsic_matrix[0, 2]  # Principal point x-coordinate
        cy = self.intrinsic_matrix[1, 2]  # Principal point y-coordinate

        # Create a grid of (x, y) coordinates
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        xv, yv = np.meshgrid(x, y)
        zv = depth_img  # Depth values

        # Convert depth image to 3D points
        x = (xv - cx) * zv / fx
        y = (yv - cy) * zv / fy
        z = zv

        # Stack the x, y, z coordinates into a single array
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        return points

    def update_point_cloud(self, depth_img):
        # Update the point cloud with new depth image
        points = self.depth2pointcloud(depth_img)  # Convert depth image to point cloud
        self.pcd.points = o3d.utility.Vector3dVector(points)  # Update the point cloud object
        self.vis.update_geometry(self.pcd)  # Update the visualizer with the new point cloud
        self.vis.poll_events()  # Process GUI events for the visualizer
        self.vis.update_renderer()  # Update the rendering

    def run(self, depth_img_generator):
        # Run the point cloud generator with a depth image generator
        try:
            while True:
                depth_img = next(depth_img_generator)  # Get the next depth image
                self.update_point_cloud(depth_img)  # Update the point cloud
                time.sleep(0.1)  # Adjust the sleep time as needed for real-time updates
        except StopIteration:
            pass  # Stop if the generator is exhausted
        finally:
            self.vis.destroy_window()  # Destroy the visualizer window

    def save_pc(self, filename):
        # Save the current point cloud to a file
        o3d.io.write_point_cloud(filename, self.pcd)