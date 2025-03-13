import vtk

def create_cube():
    """Creates a vtk cube and returns the mapper and actor."""

    # Create a cube source
    cube = vtk.vtkCubeSource()
    cube.SetXLength(1.0)
    cube.SetYLength(1.0)
    cube.SetZLength(1.0)

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return mapper, actor

def display_cube(actor):
    """Displays the vtk cube actor."""

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.9, 0.9, 0.9)  # Light gray background

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(600, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    _, cube_actor = create_cube()
    display_cube(cube_actor)