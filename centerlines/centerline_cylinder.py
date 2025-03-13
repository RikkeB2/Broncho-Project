import vtk

# Create a cylinder
cylinder = vtk.vtkCylinderSource()
cylinder.SetRadius(10)
cylinder.SetHeight(50)
cylinder.SetResolution(50)  # Increase for smoother surface
cylinder.Update()

# Define the number of disks
num_disks = 10
disk_spacing = cylinder.GetHeight() / num_disks

# Create a cutter to slice the cylinder into disks
append_filter = vtk.vtkAppendPolyData()
for i in range(num_disks + 1):
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, -cylinder.GetHeight()/2 + i * disk_spacing, 0)
    plane.SetNormal(0, 1, 0)  # Slice perpendicular to Y-axis

    cutter = vtk.vtkCutter()
    cutter.SetInputConnection(cylinder.GetOutputPort())
    cutter.SetCutFunction(plane)
    cutter.Update()

    append_filter.AddInputData(cutter.GetOutput())

append_filter.Update()

# Create mapper and actor for the sliced disks
disk_mapper = vtk.vtkPolyDataMapper()
disk_mapper.SetInputConnection(append_filter.GetOutputPort())

disk_actor = vtk.vtkActor()
disk_actor.SetMapper(disk_mapper)
disk_actor.GetProperty().SetColor(1, 1, 1)  # White disks

# Create a central axis line (Y-axis through the disk centers)
center_line = vtk.vtkLineSource()
center_line.SetPoint1(0, -cylinder.GetHeight()/2, 0)  # Bottom center
center_line.SetPoint2(0, cylinder.GetHeight()/2, 0)   # Top center
center_line.Update()

# Mapper and actor for the central axis line
line_mapper = vtk.vtkPolyDataMapper()
line_mapper.SetInputConnection(center_line.GetOutputPort())

line_actor = vtk.vtkActor()
line_actor.SetMapper(line_mapper)
line_actor.GetProperty().SetColor(1, 0, 0)  # Red central axis

# Renderer setup
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_interactor = vtk.vtkRenderWindowInteractor()
render_interactor.SetRenderWindow(render_window)

# Add both actors to the renderer
renderer.AddActor(disk_actor)
renderer.AddActor(line_actor)
renderer.SetBackground(0.1, 0.2, 0.4)  # Dark blue background

# Start visualization
render_window.Render()
render_interactor.Start()
