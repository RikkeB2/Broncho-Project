import numpy as np
import pyvista as pv
import open3d as o3d

# Load the .vtu file using PyVista
file_path = r"C:\Users\Rikke\OneDrive - Syddansk Universitet\6. semester\Bacehlor projekt\Broncho-Project\test\output_mesh.vtp"
pv_mesh = pv.read(file_path)

# Optionally convert the PyVista mesh to an Open3D PointCloud (if needed)
vertices = np.asarray(pv_mesh.points)
faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # Convert PyVista face format to Open3D format

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)

# Estimate normals for the point cloud
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Create a mesh using the Ball Pivoting Algorithm (BPA) (optional, if needed)
radii = [0.05, 0.1, 0.2]  # Adjust radii as needed
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)

# Convert the Open3D mesh back to PyVista for further processing
vertices = np.asarray(bpa_mesh.vertices)
faces = np.asarray(bpa_mesh.triangles)
faces = np.hstack([[3] + list(face) for face in faces])  # Convert to PyVista face format
pv_mesh = pv.PolyData(vertices, faces)

# Fill holes in the PyVista mesh
filled_mesh = pv_mesh.fill_holes(1000)  # Adjust the hole size limit as needed

# Clean the mesh to remove artifacts
cleaned_mesh = filled_mesh.clean()

# Smooth the mesh for better visualization
smoothed_mesh = cleaned_mesh.smooth(n_iter=30, relaxation_factor=0.1)

# Compute distance to centerlines (Figure 4)
centerline_path = r"C:\Users\Rikke\OneDrive - Syddansk Universitet\6. semester\Bacehlor projekt\Broncho-Project\test\centerline_file.vtk"
try:
    centerline = pv.read(centerline_path)
    distance = smoothed_mesh.compute_implicit_distance(centerline)
    smoothed_mesh["Distance"] = distance
except FileNotFoundError:
    print(f"Centerline file not found: {centerline_path}")
    distance = None

# Create a radius-adaptive element mesh (Figure 5)
adaptive_mesh = smoothed_mesh.decimate_pro(0.3)  # Adjust factor as needed

# Create a PyVista plotter for visualization
plotter = pv.Plotter(shape=(1, 3), window_size=(1200, 400))

# Plot Distance to Centerlines (Figure 4)
if distance is not None:
    plotter.subplot(0, 0)
    plotter.add_mesh(
        smoothed_mesh,
        scalars="Distance",
        cmap="viridis",
        show_edges=False,
        smooth_shading=True,
    )
    plotter.add_text("Distance to Centerlines", font_size=10)

# Plot Radius Adaptive Element Mesh (Figure 5)
plotter.subplot(0, 1)
plotter.add_mesh(
    adaptive_mesh,
    color="white",
    show_edges=True,
    edge_color="black",
    smooth_shading=True,
)
plotter.add_text("Radius Adaptive Mesh", font_size=10)

# Plot Internal Wireframe (Figure 6)
plotter.subplot(0, 2)
plotter.add_mesh(
    smoothed_mesh,
    style="wireframe",
    color="white",
    line_width=0.5,
)
plotter.add_text("Internal Wireframe", font_size=10)

# Set the background and show the plot
plotter.set_background("navy")
plotter.show()