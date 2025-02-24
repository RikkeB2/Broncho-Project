import os                               # For interacting with the file system
import sys                              # For adding paths to sys.path
import argparse                         # For parsing command-line arguments
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from lib.engine.onlineSimulationBO import onlineSimulationWithNetwork as onlineSimulatorBO
from scipy.io import savemat
np.random.seed(0)

# The simulator is used to generate a path, which is then used as input to the learned model.
# The learned model then generates a new path, which is used as input to the simulator.
# This process is repeated until a satisfactory path is found.


# This function parses command-line arguments to control the scripts behaviour.
def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=bool, default=True, help='Load model from a .pth file')                                           # Load model from a .pth file
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="train_set", help='Path of dataset for training and validation')       # Path of dataset for training and validation
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="checkpoints", help='Path of trained model for saving')                    # Path of trained model for saving
    parser.add_argument('--human', action='store_true', help='AI co-pilot control with human, default: Artificial Expert Agent')                            # Flag to indicate human co-pilot control, if not set AI is used.

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    online_test_centerline_names_list = ['siliconmodel3 Centerline model'] + ['siliconmodel3 Centerline model_{}'.format(x) for x in range(1, 60)]          # List of centerline models to test
    #online_test_centerline_names_list = ['siliconmodel3BO Centerline model_{}'.format(x) for x in range(1, 20)]

    global_batch_count = 0      # Initialize global batch count
    count = 0                   # Initialize count, to track the current simulation run
        
    for online_test_centerline_name in online_test_centerline_names_list:                                                   # Iterate over the list of centerline models

        simulator = onlineSimulatorBO(args.dataset_dir, online_test_centerline_name, renderer='pyrender', training=False)   # Initialize the simulator, load specified centerline model, and set renderer to pyrender
        
        path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, originalCenterlineArray, safe_distance \
            = simulator.runManual(args)                                                                                     # Run the simulation using the learned model, returns translation, rotation, list of ratios, original centerline, and safe distance 
        
        count = count + 1
        print(count)

        mdic = {"path_trajectoryT": path_trajectoryT, "path_trajectoryR": path_trajectoryR, "path_centerline_ratio_list": path_centerline_ratio_list, "originalCenterlineArray":originalCenterlineArray}        # Create a dictionary with the simulation results
        savemat("./results/automaticGT" + str(count) + ".mat", mdic)
