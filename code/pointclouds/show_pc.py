import open3d as o3d
import numpy as np
import os

def main():
    
    # Load the NumPy file
    npy_data = np.load(r"pointclouds\intermediate_point_cloud_200.pcd.npy")

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy_data)

    # Define the output directory and file path
    output_dir = "pointclouds"
    output_file = os.path.join(output_dir, "fixed_point_cloud.pcd")

    # Save as ASCII .pcd file
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print(f"Re-saved the point cloud in ASCII format at {output_file}.")

    # Load the point cloud from the PCD file
    pcd = o3d.io.read_point_cloud(output_file)  # Use .pcd file

    # Check if the point cloud is empty
    if len(pcd.points) == 0:
        print("The point cloud is empty!")
        return

    # Display the point cloud
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
