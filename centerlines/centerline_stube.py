import vtk
import math

def create_smooth_angled_tube(tube_radius=0.3, resolution=50, bend_resolution=10):
    """Creates a tube with smooth angled bends and returns the tube actor and centerline points."""

    points = vtk.vtkPoints()
    point_list = [
        [0, 0, 0],
        [2, 2, 1],
        [4, 0, 2],
        [6, 2, 3],
        [8, 0, 4],
    ]

    smooth_point_list = []
    for i in range(len(point_list) - 1):
        start_point = point_list[i]
        end_point = point_list[i + 1]
        smooth_point_list.append(start_point)

        bend_points = []
        if i < len(point_list) - 2:
            next_point = point_list[i + 2]
            v1 = [end_point[j] - start_point[j] for j in range(3)]
            v2 = [next_point[j] - end_point[j] for j in range(3)]

            normal_vector = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]

            offset_distance = 0.5
            offset_point = [
                end_point[j] - normal_vector[j] * offset_distance for j in range(3)
            ]

            for j in range(1, bend_resolution):
                t = j / bend_resolution
                bend_point = [
                    start_point[k] * (1 - t) + offset_point[k] * t for k in range(3)
                ]
                bend_points.append(bend_point)

            for j in range(1, bend_resolution):
                t = j / bend_resolution
                bend_point = [
                    offset_point[k] * (1 - t) + end_point[k] * t for k in range(3)
                ]
                bend_points.append(bend_point)

            smooth_point_list.extend(bend_points)

    smooth_point_list.append(point_list[-1])

    for p in smooth_point_list:
        points.InsertNextPoint(p[0], p[1], p[2])

    poly_line = vtk.vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(len(smooth_point_list))
    for i in range(len(smooth_point_list)):
        poly_line.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(poly_line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputData(polydata)
    tube_filter.SetRadius(tube_radius)
    tube_filter.SetNumberOfSides(resolution)
    tube_filter.Update()

    tube_mapper = vtk.vtkPolyDataMapper()
    tube_mapper.SetInputConnection(tube_filter.GetOutputPort())

    tube_actor = vtk.vtkActor()
    tube_actor.SetMapper(tube_mapper)
    tube_actor.GetProperty().SetColor(1, 1, 1)  # White tube
    tube_actor.GetProperty().SetOpacity(0.3)  # Make the tube transparent

    return tube_actor, smooth_point_list, tube_filter.GetOutput()


def create_slices_along_centerline(centerline_points, tube_polydata):
    """Creates cross-sectional slices (disks) along the tube's centerline."""

    append_filter = vtk.vtkAppendPolyData()

    for point in centerline_points:
        plane = vtk.vtkPlane()
        plane.SetOrigin(point)
        plane.SetNormal(1, 0, 0)  # Approximate normal (adjust as needed)

        cutter = vtk.vtkCutter()
        cutter.SetInputData(tube_polydata)
        cutter.SetCutFunction(plane)
        cutter.Update()

        append_filter.AddInputData(cutter.GetOutput())

    append_filter.Update()

    slice_mapper = vtk.vtkPolyDataMapper()
    slice_mapper.SetInputConnection(append_filter.GetOutputPort())

    slice_actor = vtk.vtkActor()
    slice_actor.SetMapper(slice_mapper)
    slice_actor.GetProperty().SetColor(1, 1, 1)  # White slices
    slice_actor.GetProperty().SetOpacity(1.0)  # Ensure slices are fully visible

    return slice_actor


def create_centerline_actor(centerline_points):
    """Creates a red line actor representing the centerline of the tube."""

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
    """Displays the vtk actors (tube, slices, centerline)."""

    renderer = vtk.vtkRenderer()
    for actor in actors:
        renderer.AddActor(actor)

    renderer.SetBackground(0.8, 0.8, 0.8)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    tube_actor, centerline_points, tube_polydata = create_smooth_angled_tube()
    slice_actor = create_slices_along_centerline(centerline_points, tube_polydata)
    centerline_actor = create_centerline_actor(centerline_points)

    display_actors([tube_actor, slice_actor, centerline_actor])
