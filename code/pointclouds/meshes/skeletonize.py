import trimesh
import numpy as np
import open3d as o3d

# 1. Load the mesh
mesh_path = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\intermediate_point_cloud_250.obj"
mesh = trimesh.load(mesh_path)

# 2. Calculate target_reduction
target_face_count = 1000  # Desired number of faces
original_face_count = len(mesh.faces)
target_reduction = 1 - (target_face_count / original_face_count)

# 3. Decimate the mesh
mesh_decimated = mesh.simplify_quadric_decimation(target_reduction)

# 4. Calculate Bounding Box
bounds = mesh_decimated.bounds
min_bound = bounds[0]
max_bound = bounds[1]

# 5. Approximate Centerline
center = mesh_decimated.centroid
extents = max_bound - min_bound
longest_axis = np.argmax(extents)

p1 = min_bound.copy()
p2 = max_bound.copy()

# 6. Sample Points Along Centerline
num_samples = 100  # Adjust as needed
points = np.linspace(p1, p2, num_samples)

# 7. Check if Points Are Inside the Mesh
inside_points = []
for point in points:
    if mesh_decimated.contains([point]):
        inside_points.append(point)

# 8. Visualize with Open3D
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_decimated.vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_decimated.faces)
o3d_mesh.compute_vertex_normals()

# Create a point cloud for the centerline
centerline_points = o3d.geometry.PointCloud()

if inside_points:  # Check if inside_points is not empty
    centerline_points.points = o3d.utility.Vector3dVector(np.array(inside_points))
    centerline_points.paint_uniform_color([1, 0, 0])  # Red
else:
    print("Warning: No points found inside the mesh.")

# Create a wireframe visualization (using LineSet)
lines = []
for face in mesh_decimated.faces:
    lines.extend([[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]])

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(mesh_decimated.vertices),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.paint_uniform_color([0.7, 0.7, 0.9])  # Light blue

o3d.visualization.draw_geometries([line_set, centerline_points])