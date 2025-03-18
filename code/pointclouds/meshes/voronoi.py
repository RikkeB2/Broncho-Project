import vtk

def generate_volumetric_representation(obj_file, output_file="volumetric_representation.vtk", voxel_size=0.01):
    """
    Generates a volumetric representation of a 3D object from an OBJ file and visualizes it.

    Args:
        obj_file (str): Path to the input OBJ file.
        output_file (str): Path to save the volumetric representation as a VTK file.
        voxel_size (float): Size of the voxels in the volumetric representation.
    """
    # Load the OBJ file
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update()

    polydata = reader.GetOutput()

    # Debug: Check if the OBJ file is valid
    print(f"Number of points in the OBJ file: {polydata.GetNumberOfPoints()}")
    print(f"Number of cells in the OBJ file: {polydata.GetNumberOfCells()}")

    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfCells() == 0:
        print("Error: The input OBJ file is empty or invalid.")
        return

    # Compute the bounds of the object
    bounds = polydata.GetBounds()
    print(f"Bounds of the object: {bounds}")

    # Create a voxel model
    voxel_modeller = vtk.vtkVoxelModeller()
    voxel_modeller.SetInputData(polydata)
    voxel_modeller.SetSampleDimensions(
        int((bounds[1] - bounds[0]) / voxel_size),
        int((bounds[3] - bounds[2]) / voxel_size),
        int((bounds[5] - bounds[4]) / voxel_size),
    )
    voxel_modeller.SetModelBounds(bounds)
    voxel_modeller.SetScalarTypeToFloat()
    voxel_modeller.Update()

    # Convert the voxel model to polydata
    contour_filter = vtk.vtkMarchingCubes()
    contour_filter.SetInputConnection(voxel_modeller.GetOutputPort())
    contour_filter.SetValue(0, 0.5)  # Threshold value for the isosurface
    contour_filter.Update()

    # Save the volumetric representation to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(contour_filter.GetOutput())
    writer.Write()

    print(f"Volumetric representation saved to {output_file}")

    # Extract the center structure (skeletonization)
    centerline_filter = vtk.vtkCenterlineExtraction()
    centerline_filter.SetInputData(contour_filter.GetOutput())
    centerline_filter.SetRadiusArrayName("Radius")
    centerline_filter.Update()

    # Visualize the volumetric representation and centerline
    visualize_volumetric_representation(contour_filter.GetOutput(), centerline_filter.GetOutput())


def visualize_volumetric_representation(volumetric_data, centerline_data):
    """
    Visualizes the volumetric representation and its centerline.

    Args:
        volumetric_data: The volumetric representation as vtkPolyData.
        centerline_data: The centerline representation as vtkPolyData.
    """
    # Create a mapper and actor for the volumetric data
    volume_mapper = vtk.vtkPolyDataMapper()
    volume_mapper.SetInputData(volumetric_data)

    volume_actor = vtk.vtkActor()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.GetProperty().SetOpacity(0.3)  # Make the volume semi-transparent

    # Create a mapper and actor for the centerline
    centerline_mapper = vtk.vtkPolyDataMapper()
    centerline_mapper.SetInputData(centerline_data)

    centerline_actor = vtk.vtkActor()
    centerline_actor.SetMapper(centerline_mapper)
    centerline_actor.GetProperty().SetColor(1, 0, 0)  # Red color for the centerline
    centerline_actor.GetProperty().SetLineWidth(3)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actors to the renderer
    renderer.AddActor(volume_actor)
    renderer.AddActor(centerline_actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # Background color

    # Render and start interaction
    render_window.Render()
    render_window_interactor.Start()


# Example usage
if __name__ == "__main__":
    obj_file = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\intermediate_point_cloud_100.obj"
    output_file = r"C:\Users\Lenovo\OneDrive - Syddansk Universitet\Dokumenter\GitHub\Broncho-Project\code\pointclouds\meshes\volumetric_representation.vtk"
    voxel_size = 0.5 # Adjust the voxel size as needed
    generate_volumetric_representation(obj_file, output_file, voxel_size)