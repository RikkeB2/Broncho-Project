import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np

class CenterlineProcessor:
    def __init__(self, centerline_model_dir, R_model, t_model):
        self.centerline_model_dir = centerline_model_dir
        self.R_model = R_model
        self.t_model = t_model
        self.original_centerline = None
        self.smoothed_centerline = None
        self.centerline_length = 0
        self.timestamps = None  # Initialize timestamps
        self.time_step = 1. / 120.  # Set the simulation time step
        self.load_and_process_centerline()  # Ensure this is called after initializing all attributes

    def load_and_process_centerline(self):
        # Load centerline from OBJ file
        reader = vtk.vtkOBJReader()
        reader.SetFileName(self.centerline_model_dir)
        reader.Update()

        mesh = reader.GetOutput()
        points = mesh.GetPoints()
        data = points.GetData()
        centerlineArray = vtk_to_numpy(data)
        centerlineArray = np.dot(self.R_model, centerlineArray.T).T * 0.01 + self.t_model

        # Downsample or upsample the centerline
        centerline_length = 0
        for i in range(len(centerlineArray) - 1):
            length_diff = np.linalg.norm(centerlineArray[i] - centerlineArray[i + 1])
            centerline_length += length_diff
        centerline_size = len(centerlineArray)
        lenth_size_rate = 0.007  # refer to Siliconmodel1
        centerline_size_exp = int(centerline_length / lenth_size_rate)
        centerlineArray_exp = np.zeros((centerline_size_exp, 3))
        for index_exp in range(centerline_size_exp):
            index = index_exp / (centerline_size_exp - 1) * (centerline_size - 1)
            index_left_bound = int(index)
            index_right_bound = int(index) + 1
            if index_left_bound == centerline_size - 1:
                centerlineArray_exp[index_exp] = centerlineArray[index_left_bound]
            else:
                centerlineArray_exp[index_exp] = (index_right_bound - index) * centerlineArray[index_left_bound] + (index - index_left_bound) * centerlineArray[index_right_bound]
        centerlineArray = centerlineArray_exp

        # Assign timestamps to each point
        self.assign_timestamps(centerlineArray)

        # Smoothing trajectory
        self.original_centerline = centerlineArray
        centerlineArray_smoothed = np.zeros_like(centerlineArray)
        for i in range(len(centerlineArray)):
            left_bound = i - 10
            right_bound = i + 10
            if left_bound < 0:
                left_bound = 0
            if right_bound > len(centerlineArray):
                right_bound = len(centerlineArray)
            centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound:right_bound], axis=0)
        self.smoothed_centerline = centerlineArray_smoothed

        # Calculate trajectory length
        centerline_length = 0
        for i in range(len(self.smoothed_centerline) - 1):
            length_diff = np.linalg.norm(self.smoothed_centerline[i] - self.smoothed_centerline[i + 1])
            centerline_length += length_diff
        self.centerline_length = centerline_length

    def assign_timestamps(self, centerlineArray):
        """
        Assign timestamps to each point in the centerline based on the simulation time step.
        Each point is separated by the same time step.
        """
        if self.time_step is None:
            raise ValueError("time_step is not initialized.")
        self.timestamps = np.arange(0, len(centerlineArray) * self.time_step, self.time_step)
        # Ensure the timestamps array matches the number of points
        self.timestamps = self.timestamps[:len(centerlineArray)]

    def limit_velocity(self, centerlineArray):
        limited_centerline = [centerlineArray[0]]
        for i in range(1, len(centerlineArray)):
            delta_pos = centerlineArray[i] - limited_centerline[-1]
            distance = np.linalg.norm(delta_pos)
            if distance > self.max_velocity * self.time_step:
                delta_pos = (delta_pos / distance) * (self.max_velocity * self.time_step)
            limited_centerline.append(limited_centerline[-1] + delta_pos)
        return np.array(limited_centerline)
    
    def get_timestamps(self):
        return self.timestamps

    def get_time_step(self):
        return self.time_step

    def get_original_centerline(self):
        return self.original_centerline

    def get_smoothed_centerline(self):
        return self.smoothed_centerline

    def get_centerline_length(self):
        return self.centerline_length