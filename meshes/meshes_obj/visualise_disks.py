import vtk

def visualize_segmented_disks(vtk_file):
    """
    Visualizes the segmented disks stored in a VTK file.

    Args:
        vtk_file: The path to the VTK file containing the segmented disks.
    """
    # Load the segmented disks
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()

    polydata = reader.GetOutput()

    # Debug: Check if the file contains valid data
    print(f"Number of points in the file: {polydata.GetNumberOfPoints()}")
    print(f"Number of cells in the file: {polydata.GetNumberOfCells()}")

    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfCells() == 0:
        print("Error: The VTK file contains no valid data.")
        return

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # Background color

    # Render and start interaction
    render_window.Render()
    render_window_interactor.Start()

# Example usage
if __name__ == "__main__":
    vtk_file = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\meshes\meshes_obj\segmented_disks.vtk"
    visualize_segmented_disks(vtk_file)