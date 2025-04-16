import open3d as o3d
import numpy as np
import os

def main():
    
    # Load the first NumPy file
    npy_data1 = np.load(r"C:\Users\Rikke\OneDrive - Syddansk Universitet\6. semester\Bacehlor projekt\Broncho-Project\code\pointclouds\intermediate_pointclouds\intermediate_point_cloud_100.npy")

    # Convert to Open3D point cloud
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(npy_data1)

    # Define the output directory and file path for the first point cloud
    output_dir = "pointclouds"
    output_file1 = os.path.join(output_dir, "fixed_point_cloud.pcd")

    # Save as ASCII .pcd file
    o3d.io.write_point_cloud(output_file1, pcd1, write_ascii=True)
    print(f"Re-saved the first point cloud in ASCII format at {output_file1}.")

    # Load the first point cloud from the PCD file
    pcd1 = o3d.io.read_point_cloud(output_file1)  # Use .pcd file

    # Check if the first point cloud is empty
    if len(pcd1.points) == 0:
        print("The first point cloud is empty!")
        return

    # Load the second NumPy file (replace with your actual file)
    npy_data2 = np.load(r"C:\Users\Rikke\OneDrive - Syddansk Universitet\6. semester\Bacehlor projekt\Broncho-Project\code\pointclouds\intermediate_pointclouds\intermediate_point_cloud_300.npy")

    # Convert to Open3D point cloud
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(npy_data2)

    # Define the output directory and file path for the second point cloud
    output_file2 = os.path.join(output_dir, "fixed_point_cloud2.pcd")

    # Save as ASCII .pcd file
    o3d.io.write_point_cloud(output_file2, pcd2, write_ascii=True)
    print(f"Re-saved the second point cloud in ASCII format at {output_file2}.")

    # Load the second point cloud from the PCD file
    pcd2 = o3d.io.read_point_cloud(output_file2)  # Use .pcd file

    # Check if the second point cloud is empty
    if len(pcd2.points) == 0:
        print("The second point cloud is empty!")
        return

    # Display the first point cloud in a separate window
    print(f"Loaded first point cloud with {len(pcd1.points)} points.")
    o3d.visualization.draw_geometries([pcd1])

    # Display the second point cloud in another separate window
    print(f"Loaded second point cloud with {len(pcd2.points)} points.")
    o3d.visualization.draw_geometries([pcd2])

if __name__ == "__main__":
    main()