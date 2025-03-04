import open3d as o3d
import numpy as np

def main():

    # Load the NumPy file
    npy_data = np.load(r"C:\Users\Rikke\OneDrive - Syddansk Universitet\6. semester\Bacehlor projekt\Broncho-Project\code\pointclouds\final_point_cloud.pcd.npy")

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy_data)

    # Save as ASCII .pcd file
    o3d.io.write_point_cloud("fixed_point_cloud.pcd", pcd, write_ascii=True)
    print("Re-saved the point cloud in ASCII format.")

    # Load the point cloud from the PCD file
    pcd = o3d.io.read_point_cloud("fixed_point_cloud.pcd")  # Use .pcd file

    # Check if the point cloud is empty
    if len(pcd.points) == 0:
        print("The point cloud is empty!")
        return

    # Display the point cloud
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
