import vtk

def ply_to_vtp(input_ply, output_vtp):
    """
    Converts a .ply file to a .vtp file.

    Args:
        input_ply (str): Path to the input .ply file.
        output_vtp (str): Path to save the output .vtp file.
    """
    reader = vtk.vtkPLYReader()
    reader.SetFileName(input_ply)
    reader.Update()

    polydata = reader.GetOutput()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp)
    writer.SetInputData(polydata)
    writer.Write()

if __name__ == '__main__':
    input_file = "input.ply"  # Replace with your input .ply file
    output_file = "output.vtp" # Replace with your desired output .vtp file
    ply_to_vtp(input_file, output_file)
    print(f"Successfully converted '{input_file}' to '{output_file}'")