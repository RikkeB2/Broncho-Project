import vtk
import math

def create_smooth_angled_tube(tube_radius=0.3, resolution=50, bend_resolution=10):
    """Creates a tube with smooth angled bends."""

    points = vtk.vtkPoints()
    point_list = [
        [0, 0, 0],
        [2, 2, 1],
        [4, 0, 2],
        [6, 2, 3],
        [8, 0, 4],
    ]

    smooth_point_list =[]
    for i in range(len(point_list) - 1):
        start_point = point_list[i]
        end_point = point_list[i + 1]

        # Add the starting point
        smooth_point_list.append(start_point)

        # Calculate bend points
        bend_points =[]
        if i < len(point_list) - 2:
            next_point = point_list[i + 2]
            v1 = [end_point[0] - start_point[0], end_point[1] - start_point[1], end_point[2] - start_point[2]]
            v2 = [next_point[0] - end_point[0], next_point[1] - end_point[1], next_point[2] - end_point[2]]

            # Calculate a vector orthogonal to the plane of the bend
            normal_vector = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]

            # Calculate a point offset from the original point
            offset_distance = 0.5  # Adjust this for bend size
            offset_point = [
                end_point[0] - normal_vector[0] * offset_distance,
                end_point[1] - normal_vector[1] * offset_distance,
                end_point[2] - normal_vector[2] * offset_distance,
            ]

            # Create smooth bend points
            for j in range(1, bend_resolution):
                t = j / bend_resolution
                bend_point = [
                    start_point[0] * (1 - t) + offset_point[0] * t,
                    start_point[1] * (1 - t) + offset_point[1] * t,
                    start_point[2] * (1 - t) + offset_point[2] * t,
                ]
                bend_points.append(bend_point)

            for j in range(1, bend_resolution):
                t = j / bend_resolution
                bend_point = [
                    offset_point[0] * (1 - t) + end_point[0] * t,
                    offset_point[1] * (1 - t) + end_point[1] * t,
                    offset_point[2] * (1 - t) + end_point[2] * t,
                ]
                bend_points.append(bend_point)

            smooth_point_list.extend(bend_points)

    # Add the final point
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

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def display_actor(actor):
    """Displays the vtk actor."""

    renderer = vtk.vtkRenderer()
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
    smooth_tube_actor = create_smooth_angled_tube()
    display_actor(smooth_tube_actor)