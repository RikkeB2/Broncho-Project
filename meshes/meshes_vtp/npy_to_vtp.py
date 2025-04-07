import os
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

# === CONFIG ===
input_folder = "code\pointclouds\intermediate_pointclouds"
output_folder = "code\pointclouds\meshes"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through .npy files
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        filepath = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, base_name + ".vtp")

        # Load point cloud
        points_array = np.load(filepath)

        # Check shape
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            print(f"Skipping {filename}: not a (N, 3) array.")
            continue

        # Convert to VTK points
        vtk_points = vtk.vtkPoints()
        vtk_array = numpy_to_vtk(points_array, deep=True)
        vtk_points.SetData(vtk_array)

        # Create PolyData and assign points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # Write to .vtp
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(polydata)
        writer.Write()

        print(f"Converted: {filename} → {output_path}")
