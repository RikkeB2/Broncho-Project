import open3d as o3d

def visualize_obj(file_path):
    """
    Loads and visualizes an .obj file using Open3D.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    if mesh.is_empty():
        print("Error: Unable to load the mesh. Check the file path.")
        return

    # Compute normals for better shading
    mesh.compute_vertex_normals()

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    file_path = "code\pointclouds\meshes\intermediate_point_cloud_260.obj"  # Change this to OBJ file path
    visualize_obj(file_path)