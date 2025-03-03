import open3d as o3d

def main():
    # Load the point cloud from the file
    pcd = o3d.io.read_point_cloud("intermediate_point_cloud_10.pcd") # change to the correct file

    # Check if the point cloud is empty
    if len(pcd.points) == 0:
        print("The point cloud is empty!")
        return

    # Display the point cloud
    print("Displaying the point cloud...")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()