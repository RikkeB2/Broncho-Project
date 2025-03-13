import open3d as o3d
import numpy as np
import os
import time  # Import the time module

def convert_pcd_to_obj(input_dir, output_dir):
    """
    Converts all PCD files in a directory to OBJ meshes using ball pivoting.

    Args:
        input_dir (str): Path to the directory containing the PCD files.
        output_dir (str): Path to the directory where OBJ files will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pcd") and filename.startswith("intermediate_point_cloud_"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename[:-4] + ".obj")  # Replace .pcd with .obj

            try:
                start_time = time.time()  # Start timer

                # Load point cloud
                print(f"Loading point cloud: {filename}")  # Added loading message
                pcd = o3d.io.read_point_cloud(input_path)

                # Estimate normals (important for meshing)
                print(f"Estimating normals for: {filename}") # Added normals message
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.])) # added to orient normals

                # Ball pivoting meshing
                print(f"Creating mesh for: {filename}") # Added meshing message
                radii = [0.005, 0.01, 0.02, 0.04] #adjust radii as needed
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))


                # Save mesh
                print(f"Saving mesh for: {filename}") # Added saving message
                o3d.io.write_triangle_mesh(output_path, mesh)

                end_time = time.time()  # End timer
                elapsed_time = end_time - start_time
                print(f"Converted {filename} to {filename[:-4]}.obj in {elapsed_time:.2f} seconds")

            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    input_directory = r"c:\Users\Rikke\OneDrive - Syddansk Universitet\6. semester\Bacehlor projekt\Broncho-Project\code\pointclouds"  # Replace with the actual path to your PCD files
    output_directory = r"c:\Users\Rikke\OneDrive - Syddansk Universitet\6. semester\Bacehlor projekt\Broncho-Project\code\pointclouds\meshes" # Replace with the desired output directory
    convert_pcd_to_obj(input_directory, output_directory)