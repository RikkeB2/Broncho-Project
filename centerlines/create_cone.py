import vtk

def create_cone(
    radius=1.0,  # Radius of the base
    height=2.0,  # Height of the cone
    resolution=50,  # Number of sides for the base (smoothness)
):
    """Creates a vtk hollow cone and returns the actor."""

    # Create a cone source
    cone = vtk.vtkConeSource()
    cone.SetRadius(radius)
    cone.SetHeight(height)
    cone.SetResolution(resolution)

    # Extract the surface
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(cone.GetOutputPort())
    geometry_filter.Update()

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(geometry_filter.GetOutput())

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def display_actor(actor):
    """Displays the vtk actor."""

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.8, 0.8, 0.8)  # Light gray background

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window.SetSize(600, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    cone_actor = create_cone()
    display_actor(cone_actor)