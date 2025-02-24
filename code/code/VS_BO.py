import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from lib.engine.onlineSimulationBO import onlineSimulationWithNetwork as onlineSimulatorBO
from scipy.io import savemat
np.random.seed(0)


# Code is very similar to manualBO, but uses runVS2 method, which returns additional data about joints.

# Parses command-line arguments to control the scripts behaviour.
def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=bool, default=True, help='Load model from a .pth file')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="train_set", help='Path of dataset for training and validation')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="checkpoints", help='Path of trained model for saving')
    parser.add_argument('--human', action='store_true', help='AI co-pilot control with human, default: Artificial Expert Agent')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    online_test_centerline_names_list = ['siliconmodel3 Centerline model'] + ['siliconmodel3 Centerline model_{}'.format(x) for x in range(1, 60)]  # List of centerline models to test
    #online_test_centerline_names_list = ['siliconmodel3BO Centerline model_{}'.format(x) for x in range(1, 20)]

    global_batch_count = 0  # Initialize global batch count
    count = 0               # Initialize count, to track the current simulation run
        
    for online_test_centerline_name in online_test_centerline_names_list:                                                                   # Iterate over the list of centerline models
        simulator = onlineSimulatorBO(args.dataset_dir, online_test_centerline_name, renderer='pyrender', training=False)                   # Initialize the simulator, load specified centerline model, and set renderer to pyrender
        path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, originalCenterlineArray, safe_distance, path_jointvel, path_joint \
            = simulator.runVS2(args)                                                                                                        # Run the simulation using the learned model, returns translation, rotation, list of ratios, original centerline, safe distance, joint velocities, and joint angles       
        
        count = count + 1
        print(count)

        mdic = {"path_trajectoryT": path_trajectoryT, "path_trajectoryR": path_trajectoryR, "path_centerline_ratio_list": path_centerline_ratio_list, \
                "originalCenterlineArray":originalCenterlineArray, "path_jointvel":path_jointvel, "path_joint":path_joint}
        savemat("./results/vsGT" + str(count) + ".mat", mdic)
