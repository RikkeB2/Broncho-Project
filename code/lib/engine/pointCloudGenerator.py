import open3d as o3d
import numpy as np
import time

class PointCloudGenerator:
    def __init__(self, intrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

    def depth2pointcloud(self, depth_img):
        H, W = depth_img.shape
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]

        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        xv, yv = np.meshgrid(x, y)
        zv = depth_img

        x = (xv - cx) * zv / fx
        y = (yv - cy) * zv / fy
        z = zv

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        return points

    def update_point_cloud(self, depth_img):
        # Update the point cloud with new depth image
        points = self.depth2pointcloud(depth_img)
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self, depth_img_generator):
        try:
            while True:
                depth_img = next(depth_img_generator)
                self.update_point_cloud(depth_img)
                time.sleep(0.1)  # Adjust the sleep time as needed for real-time updates
        except StopIteration:
            pass
        finally:
            self.vis.destroy_window()