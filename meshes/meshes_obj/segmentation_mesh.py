import vtk

def segment_mesh_into_disks(mesh_file, num_disks, output_file="segmented_disks.vtk"):
    """
    Splits a 3D mesh into horizontal closed disks and saves all disks into a single VTK file.
    Uses vtkDelaunay2D for triangulation.
    """
    reader = vtk.vtkOBJReader()
    reader.SetFileName(mesh_file)
    reader.Update()

    mesh = reader.GetOutput()

    print(f"Number of points in the mesh: {mesh.GetNumberOfPoints()}")
    print(f"Number of cells in the mesh: {mesh.GetNumberOfCells()}")

    if mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0:
        print("Error: The input mesh is empty or invalid.")
        return

    bounds = mesh.GetBounds()
    z_min, z_max = bounds[4], bounds[5]
    disk_spacing = (z_max - z_min) / num_disks

    append_filter = vtk.vtkAppendPolyData()  # Initialize append filter

    for i in range(num_disks + 1):
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, z_min + i * disk_spacing)
        plane.SetNormal(0, 0, 1)

        cutter = vtk.vtkCutter()
        cutter.SetInputData(mesh)
        cutter.SetCutFunction(plane)
        cutter.Update()

        print(f"Disk {i}: Number of points after cutting: {cutter.GetOutput().GetNumberOfPoints()}")
        print(f"Disk {i}: Number of cells after cutting: {cutter.GetOutput().GetNumberOfCells()}")

        if cutter.GetOutput().GetNumberOfPoints() == 0:
            print(f"Warning: Disk {i} has no points and will be skipped.")
            continue

        # Ensure closed loops using connectivity filter
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(cutter.GetOutput())
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()

        num_regions = connectivity.GetNumberOfExtractedRegions()
        print(f"Disk {i}: Number of regions after connectivity: {num_regions}")

        stripper = vtk.vtkStripper()
        stripper.SetInputData(connectivity.GetOutput())
        stripper.Update()

        if stripper.GetOutput().GetNumberOfCells() == 0:
            print(f"Warning: Disk {i} does not form a closed loop and will be skipped.")
            continue

        # Triangulate using vtkDelaunay2D
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(stripper.GetOutput())
        delaunay.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(delaunay.GetOutput())
        cleaner.Update()

        append_filter.AddInputData(cleaner.GetOutput())  # Append the disk data

    append_filter.Update()  # Update the append filter after all disks are added

    # Save all disks to a single file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(append_filter.GetOutput())
    writer.Write()

    print(f"All segmented disks saved to {output_file}")

if __name__ == "__main__":
    mesh_file = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\intermediate_point_cloud_100.obj"
    num_disks = 100
    segment_mesh_into_disks(mesh_file, num_disks)