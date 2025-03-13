import vtk

def create_cylinder():
    """Creates a vtk cylinder and returns the mapper and actor."""

    # Create a cylinder source
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetHeight(3.0)
    cylinder.SetRadius(1.0)
    cylinder.SetResolution(100)  # Increase resolution for smoother cylinder

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cylinder.GetOutputPort())

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return mapper, actor

def display_cylinder(actor):
    """Displays the vtk cylinder actor."""

    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # Dark blue background

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(600, 600)

    # Create an interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Initialize and start the interactor
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    _, cylinder_actor = create_cylinder()
    display_cylinder(cylinder_actor)