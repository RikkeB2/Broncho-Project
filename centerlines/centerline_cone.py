import vtk


def create_cone(radius=1.0, height=2.0, resolution=50):
    """Creates a semi-transparent cone and returns the actor + centerline points."""

    cone = vtk.vtkConeSource()
    cone.SetRadius(radius)
    cone.SetHeight(height)
    cone.SetResolution(resolution)
    cone.SetDirection(0, 0, 1)  # Ensure it points upwards
    cone.SetCenter(0, 0, height / 2)  # Center at origin

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(cone.GetOutputPort())
    geometry_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(geometry_filter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 1, 1)  # White cone
    actor.GetProperty().SetOpacity(0.3)  # Semi-transparent for visibility

    # Generate centerline points along the cone's axis
    centerline_points = [[0, 0, z] for z in range(0, int(height) + 1)]

    return actor, centerline_points, geometry_filter.GetOutput()


def create_slices_along_centerline(centerline_points, cone_polydata):
    """Creates cross-sectional slices (disks) along the cone's height."""

    append_filter = vtk.vtkAppendPolyData()

    for point in centerline_points:
        plane = vtk.vtkPlane()
        plane.SetOrigin(point)
        plane.SetNormal(0, 0, 1)  # Horizontal slices

        cutter = vtk.vtkCutter()
        cutter.SetInputData(cone_polydata)
        cutter.SetCutFunction(plane)
        cutter.Update()

        append_filter.AddInputData(cutter.GetOutput())

    append_filter.Update()

    slice_mapper = vtk.vtkPolyDataMapper()
    slice_mapper.SetInputConnection(append_filter.GetOutputPort())

    slice_actor = vtk.vtkActor()
    slice_actor.SetMapper(slice_mapper)
    slice_actor.GetProperty().SetColor(1, 1, 1)  # White slices
    slice_actor.GetProperty().SetOpacity(1.0)  # Fully visible

    return slice_actor


def create_centerline_actor(centerline_points):
    """Creates a red line actor representing the centerline of the cone."""

    centerline_polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    line = vtk.vtkPolyLine()

    for i, p in enumerate(centerline_points):
        points.InsertNextPoint(p)
        line.GetPointIds().InsertNextId(i)

    lines.InsertNextCell(line)
    centerline_polydata.SetPoints(points)
    centerline_polydata.SetLines(lines)

    centerline_mapper = vtk.vtkPolyDataMapper()
    centerline_mapper.SetInputData(centerline_polydata)

    centerline_actor = vtk.vtkActor()
    centerline_actor.SetMapper(centerline_mapper)
    centerline_actor.GetProperty().SetColor(1, 0, 0)  # Red centerline
    centerline_actor.GetProperty().SetLineWidth(3)  # Make it more visible

    return centerline_actor


def display_actors(actors):
    """Displays the vtk actors (cone, slices, centerline)."""

    renderer = vtk.vtkRenderer()
    for actor in actors:
        renderer.AddActor(actor)

    renderer.SetBackground(0.8, 0.8, 0.8)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(600, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    cone_actor, centerline_points, cone_polydata = create_cone()
    slice_actor = create_slices_along_centerline(centerline_points, cone_polydata)
    centerline_actor = create_centerline_actor(centerline_points)

    display_actors([cone_actor, slice_actor, centerline_actor])
