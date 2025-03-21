import open3d as o3d
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

def extract_centerline(mesh_path, num_points=5000, k_neighbors=10):
    """
    Extracts a centerline from a mesh using Open3D and NetworkX.

    Args:
        mesh_path (str): Path to the mesh file (e.g., .obj, .ply).
        num_points (int): Number of points to sample from the mesh.
        k_neighbors (int): Number of nearest neighbors to connect in the graph.

    Returns:
        tuple: (open3d.geometry.PointCloud, open3d.geometry.LineSet) - The sampled point cloud and the extracted centerline.
    """

    # Load mesh and convert to point cloud
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(num_points)
    points = np.asarray(pcd.points)

    # Build KDTree and find nearest neighbors
    tree = KDTree(points)
    _, indices = tree.query(points, k=k_neighbors)

    # Build graph from point cloud
    graph = nx.Graph()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Ignore self-connection
            distance = np.linalg.norm(points[i] - points[neighbor])
            graph.add_edge(i, neighbor, weight=distance)

    # Find start and end points (lowest and highest Z)
    start_idx = np.argmin(points[:, 2])
    end_idx = np.argmax(points[:, 2])

    # Debugging: Print start and end points
    print(f"Start point index: {start_idx}, coordinates: {points[start_idx]}")
    print(f"End point index: {end_idx}, coordinates: {points[end_idx]}")

    # Debugging: Check if graph is connected
    if not nx.is_connected(graph):
        print("Graph is disconnected.")
        components = list(nx.connected_components(graph))
        print(f"Number of connected components: {len(components)}")

    # Compute shortest path and centerline points
    try:
        shortest_path = nx.shortest_path(graph, source=start_idx, target=end_idx, weight="weight")
        centerline_points = points[shortest_path]

        # Create and color the centerline LineSet
        centerline = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(centerline_points),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centerline_points) - 1)])
        )
        centerline.paint_uniform_color([1, 0, 0])  # Red

        return pcd, centerline

    except nx.NetworkXNoPath:
        print(f"No path between {start_idx} and {end_idx}. Try increasing k_neighbors or checking start/end points.")
        return pcd, None #Return None centerline, so that the program doesn't crash.

if __name__ == "__main__":
    mesh_path = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\intermediate_point_cloud_250.obj"  # Replace with your mesh path
    pcd, centerline = extract_centerline(mesh_path)
    if centerline is not None:
        o3d.visualization.draw_geometries([pcd, centerline])
    else:
        o3d.visualization.draw_geometries([pcd]) #Show only the pointcloud if centerline creation failed.