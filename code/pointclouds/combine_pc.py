import open3d as o3d
import numpy as np
import os
import re

def create_pose_graph_from_translation_log(log_file):
    """
    Creates an Open3D pose graph from a log file containing only translation data.

    Args:
        log_file (str): Path to the log file.

    Returns:
        o3d.pipelines.registration.PoseGraph: The created pose graph.
    """

    pose_graph = o3d.pipelines.registration.PoseGraph()
    node_id = 0  # Initialize node ID

    with open(log_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                # Use regular expression to extract translation
                match = re.match(r'Translation:\s*(\[[\d\.\s+-]+(?:,\s*[\d\.\s+-]+)*\])', line)
                if match:
                    translation_str = match.group(1)

                    # Convert translation string to numpy array
                    translation = np.fromstring(translation_str.strip('[]'), sep=',')

                    # Create 4x4 transformation matrix (identity rotation)
                    transformation = np.eye(4)
                    transformation[:3, 3] = translation

                    # Add node to pose graph
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(transformation))

                    # Add edge (connecting nodes) - you might need to adjust this based on your data
                    if node_id > 0:
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(node_id - 1, node_id, transformation,
                                                                     information=np.identity(6), uncertain=False)
                        )
                    node_id += 1  # Increment node ID
                else:
                    print(f"Warning: Could not parse line: {line.strip()}")

            except Exception as e:
                print(f"Error processing line: {line.strip()}. Error: {e}")

    return pose_graph

def combine_point_clouds(directory="pointclouds", pose_graph=None):
    """
    Combines all PCD files in a directory into a single point cloud,
    applying transformations from a pose graph if provided.

    Args:
        directory (str): The directory containing the PCD files.
        pose_graph (o3d.pipelines.registration.PoseGraph, optional): A pose graph
            containing transformations for each point cloud. Defaults to None.
    """

    point_clouds = []
    filenames = []  # Keep track of filenames for pose graph lookup
    for filename in os.listdir(directory):
        if filename.endswith(".pcd") and filename.startswith("intermediate_point_cloud_"):
            filepath = os.path.join(directory, filename)
            try:
                pcd = o3d.io.read_point_cloud(filepath)
                point_clouds.append(pcd)
                filenames.append(filename)  # Store filename
                print(f"Successfully loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not point_clouds:
        print("No PCD files found in the specified directory.")
        return None

    # Combine all point clouds
    combined_point_cloud = o3d.geometry.PointCloud()

    if pose_graph is not None:
        # Apply transformations from pose graph
        num_nodes = len(pose_graph.nodes)
        num_point_clouds = len(point_clouds)

        # Check if the number of nodes in the pose graph matches the number of point clouds
        if num_nodes != num_point_clouds:
            print(f"Warning: Number of nodes in pose graph ({num_nodes}) does not match number of point clouds ({num_point_clouds}).")
            min_len = min(num_nodes, num_point_clouds)  # Use the smaller length
        else:
            min_len = num_point_clouds  # Use the number of point clouds

        for i in range(min_len):
            pcd.transform(pose_graph.nodes[i].pose)
            combined_point_cloud += pcd
    else:
        # No transformations, just combine
        for pcd in point_clouds:
            combined_point_cloud += pcd

    return combined_point_cloud

if __name__ == "__main__":
    print("Combining intermediate point clouds...")

    # Create pose graph from translation log file
    pose_graph = create_pose_graph_from_translation_log("translation_log.txt")

    combined_pc = combine_point_clouds(directory="pointclouds", pose_graph=pose_graph) # Pass the pose_graph

    if combined_pc is not None:
        # Save the combined point cloud as PCD
        o3d.io.write_point_cloud("combined_point_cloud.pcd", combined_pc)
        print("Combined point cloud saved as combined_point_cloud.pcd")

        # Convert to numpy array
        points = np.asarray(combined_pc.points)

        # Save the combined point cloud as numpy array
        np.save("combined_point_cloud.npy", points)
        print("Combined point cloud saved as combined_point_cloud.npy")

        # Visualize the combined point cloud
        print("Visualizing combined point cloud...")
        o3d.visualization.draw_geometries([combined_pc])