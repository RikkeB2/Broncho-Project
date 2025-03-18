import open3d as o3d
import numpy as np

def get_mesh_volume(mesh, voxel_size=0.01):
    """
    Get the volumetric representation (inside) of a mesh using voxelization.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        voxel_size (float): The size of the voxels for voxelization.
    
    Returns:
        np.ndarray: A binary 3D numpy array representing the inside of the mesh.
    """
    # Check if the mesh contains triangles
    if len(mesh.triangles) == 0:
        raise ValueError("The input mesh does not contain triangles. Please provide a valid mesh.")

    # Convert the mesh to a voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    print("Voxel grid created.")

    # Convert voxel grid to a binary 3D numpy array
    voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    if voxel_indices.size == 0:
        raise ValueError("Voxel grid creation failed. The mesh might be invalid or too sparse.")

    grid_shape = np.max(voxel_indices, axis=0) + 1
    binary_grid = np.zeros(grid_shape, dtype=bool)
    binary_grid[tuple(voxel_indices.T)] = True
    print("Binary grid created.")

    return binary_grid

def visualize_volume(binary_grid, voxel_size=0.01):
    """
    Visualize the volumetric representation using Open3D.
    
    Args:
        binary_grid (np.ndarray): A binary 3D numpy array representing the inside of the mesh.
        voxel_size (float): The size of the voxels for visualization.
    """
    # Get the indices of the filled voxels
    filled_voxels = np.argwhere(binary_grid)

    # Create an Open3D point cloud for visualization
    points = filled_voxels * voxel_size  # Scale voxel indices to real-world coordinates
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Visualize the volumetric representation
    o3d.visualization.draw_geometries([point_cloud])
    print("Volumetric representation visualized.")

if __name__ == "__main__":
    # Example usage
    file_path = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\accumulated_point_cloud.ply"  # Replace with your point cloud file path
    
    # Load the mesh or point cloud
    mesh = o3d.io.read_triangle_mesh(file_path)
    print("Mesh loaded.")

    # If the input is a point cloud, convert it to a mesh
    if len(mesh.triangles) == 0:
        print("Input appears to be a point cloud. Converting to a mesh using Poisson reconstruction...")
        point_cloud = o3d.io.read_point_cloud(file_path)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
        print("Mesh created from point cloud.")

    # Get volumetric representation
    volume = get_mesh_volume(mesh)
    print(f"Volumetric representation created with shape {volume.shape}.")

    # Visualize the volumetric representation
    visualize_volume(volume)
