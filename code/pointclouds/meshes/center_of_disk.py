import vtk

def center_of_disks(vtk_file, output_file="disk_centers.vtk"):
    """
    Calculates the center of each disk in a segmented VTK file and visualizes them.
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()

    polydata = reader.GetOutput()

    if polydata.GetNumberOfCells() == 0:
        print("Error: The VTK file contains no valid data.")
        return

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()

    num_regions = connectivity.GetNumberOfExtractedRegions()

    centers_points = vtk.vtkPoints()
    centers_polydata = vtk.vtkPolyData()
    centers_vertices = vtk.vtkCellArray()

    for region_id in range(num_regions):
        extracted_polydata = connectivity.GetOutput(region_id)

        if extracted_polydata is None: #Add this check.
            continue

        if extracted_polydata.GetNumberOfPoints() == 0:
            continue

        bounds = extracted_polydata.GetBounds()
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]

        centers_points.InsertNextPoint(center)
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, region_id)
        centers_vertices.InsertNextCell(vertex)

    centers_polydata.SetPoints(centers_points)
    centers_polydata.SetVerts(centers_vertices)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(centers_polydata)
    writer.Write()

    print(f"Disk centers saved to {output_file}")

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(centers_polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(10)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    vtk_file = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\segmented_disks.vtk"
    center_of_disks(vtk_file)