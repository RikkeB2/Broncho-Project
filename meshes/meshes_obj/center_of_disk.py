import vtk

def segment_mesh_into_disks(mesh_file, num_disks, output_file="segmented_disks.vtk", centers_output_file="disk_centers.vtk"):
    """Segments the mesh, filters disks, calculates centers, and saves all to VTK."""
    reader = vtk.vtkOBJReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    mesh = reader.GetOutput()
    bounds = mesh.GetBounds()
    z_min, z_max = bounds[4], bounds[5]
    disk_spacing = (z_max - z_min) / num_disks
    append_filter = vtk.vtkAppendPolyData()
    centers_points = vtk.vtkPoints()
    centers_polydata = vtk.vtkPolyData()
    centers_vertices = vtk.vtkCellArray()
    region_count = 0

    for i in range(num_disks + 1):
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, z_min + i * disk_spacing)
        plane.SetNormal(0, 0, 1)
        cutter = vtk.vtkCutter()
        cutter.SetInputData(mesh)
        cutter.SetCutFunction(plane)
        cutter.Update()

        if cutter.GetOutput().GetNumberOfPoints() == 0:
            continue

        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(cutter.GetOutput())
        connectivity.SetExtractionModeToAllRegions()
        connectivity.Update()
        num_regions = connectivity.GetNumberOfExtractedRegions()

        for region_id in range(num_regions):
            extracted_polydata = connectivity.GetOutput(region_id)

            if extracted_polydata is None or extracted_polydata.GetNumberOfPoints() == 0:
                continue

            stripper = vtk.vtkStripper()
            stripper.SetInputData(extracted_polydata)
            stripper.Update()

            if stripper.GetOutput().GetNumberOfCells() == 0:
                continue

            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(stripper.GetOutput())
            delaunay.Update()

            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(delaunay.GetOutput())
            cleaner.Update()

            if cleaner.GetOutput().GetNumberOfPoints() > 10:  # Filtering logic
                append_filter.AddInputData(cleaner.GetOutput())
                bounds = cleaner.GetOutput().GetBounds()
                center = [(bounds[0] + bounds[1]) / 2,
                          (bounds[2] + bounds[3]) / 2,
                          (bounds[4] + bounds[5]) / 2]
                centers_points.InsertNextPoint(center)
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, region_count)
                centers_vertices.InsertNextCell(vertex)
                region_count += 1

    append_filter.Update()
    centers_polydata.SetPoints(centers_points)
    centers_polydata.SetVerts(centers_vertices)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(append_filter.GetOutput())
    writer.Write()
    centers_writer = vtk.vtkPolyDataWriter()
    centers_writer.SetFileName(centers_output_file)
    centers_writer.SetInputData(centers_polydata)
    centers_writer.Write()
    print(f"Segmented disks saved to {output_file}")
    print(f"Disk centers saved to {centers_output_file}")

def visualize_segmented_disks_and_centers(vtk_file, centers_file):
    """Visualizes segmented disks, centers, and line connecting centers."""
    disks_reader = vtk.vtkPolyDataReader()
    disks_reader.SetFileName(vtk_file)
    disks_reader.Update()
    disks_polydata = disks_reader.GetOutput()

    centers_reader = vtk.vtkPolyDataReader()
    centers_reader.SetFileName(centers_file)
    centers_reader.Update()
    centers_polydata = centers_reader.GetOutput()

    # Disks: Red and very see-through
    disks_mapper = vtk.vtkPolyDataMapper()
    disks_mapper.SetInputData(disks_polydata)
    disks_actor = vtk.vtkActor()
    disks_actor.SetMapper(disks_mapper)
    disks_actor.GetProperty().SetColor(1, 0, 0)  # Red
    disks_actor.GetProperty().SetOpacity(0.1)  # Very see-through

    # Centers: Small points
    centers_mapper = vtk.vtkPolyDataMapper()
    centers_mapper.SetInputData(centers_polydata)
    centers_actor = vtk.vtkActor()
    centers_actor.SetMapper(centers_mapper)
    centers_actor.GetProperty().SetPointSize(5)  # Smaller points

    points = centers_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    if num_points > 1:
        line_points = vtk.vtkPoints()
        line_cells = vtk.vtkCellArray()
        for i in range(num_points):
            line_points.InsertNextPoint(points.GetPoint(i))
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(num_points)
        for i in range(num_points):
            line.GetPointIds().SetId(i, i)
        line_cells.InsertNextCell(line)
        line_polydata = vtk.vtkPolyData()
        line_polydata.SetPoints(line_points)
        line_polydata.SetLines(line_cells)
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputData(line_polydata)
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(0, 1, 0)  # Green line
        line_actor.GetProperty().SetLineWidth(3)  # Slightly thicker line

        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        renderer.AddActor(disks_actor)
        renderer.AddActor(centers_actor)
        renderer.AddActor(line_actor)
        renderer.SetBackground(0.8, 0.8, 0.8)  # Light gray background

        render_window.Render()
        render_window_interactor.Start()
    else:
        print("Not enough center points to draw a line.")
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        renderer.AddActor(disks_actor)
        renderer.AddActor(centers_actor)
        renderer.SetBackground(0.8, 0.8, 0.8)  # Light gray background

        render_window.Render()
        render_window_interactor.Start()

if __name__ == "__main__":
    mesh_file = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\intermediate_point_cloud_100.obj"
    num_disks = 100
    segment_mesh_into_disks(mesh_file, num_disks)
    visualize_segmented_disks_and_centers("segmented_disks.vtk", "disk_centers.vtk")