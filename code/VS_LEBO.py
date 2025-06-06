import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time

from lib.engine.onlineSimulationLEBO import onlineSimulationWithNetwork as onlineSimulatorBO
from lib.engine.pointCloudGenerator import PointCloudGenerator
from lib.engine.centerLineGenerator import CenterlineGenerator

from scipy.io import savemat

np.random.seed(0)


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

    log_dir = os.path.join(args.model_dir, './logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Set the path to the dataset directory
    time_log_path = os.path.join(log_dir, "timing_log.txt")
    time_log_file = open(time_log_path, "w")


    online_test_centerline_names_list = ['siliconmodel3 Centerline model'] + ['siliconmodel3 Centerline model_{}'.format(x) for x in range(1, 60)]
    #online_test_centerline_names_list = ['siliconmodel3BO Centerline model_{}'.format(x) for x in range(1, 20)]


    global_batch_count = 0
    count = 0

    # Ensure the pointclouds directory exists
    pointclouds_dir = "./pointclouds"
    if not os.path.exists(pointclouds_dir):
        os.makedirs(pointclouds_dir)

    # Initialize PointCloudGenerator
    intrinsic_matrix = np.array([[181.93750381, 0, 200],
                            [0, 181.93750381, 200],
                            [0, 0, 1]])  #HERE BO
     # Normalize the intrinsic matrix
    #intrinsic_matrix_normalized = intrinsic_matrix / intrinsic_matrix[2, 2]

    
    point_cloud_generator = PointCloudGenerator(intrinsic_matrix)
    centerline_generator = CenterlineGenerator(
    num_points=100,
    downsample_voxel=None,
    k_neighbors=8,
    neigh_k=20,  
    slab_factor=1.2,
    min_slice_count=30, 
    smooth_window=71, 
    smooth_polyorder=4,
    min_clearance=2.0,
    hull_area_thresh=5.0
    )

    try:
        overall_start = time.time()  # Start timing the whole simulation

        simulator = onlineSimulatorBO(args.dataset_dir, online_test_centerline_names_list[0], renderer='pyrender', training=False)
        for centerline_name in online_test_centerline_names_list:
            centerline_start = time.time()  # Start timing this centerline

            simulator.load_centerline(centerline_name)
            path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, originalCenterlineArray, safe_distance, path_jointvel, path_joint = \
                simulator.runVS2(args, point_cloud_generator, centerline_generator)

            centerline_end = time.time()
            centerline_time = centerline_end - centerline_start
            minutes, seconds = divmod(centerline_time, 60)
            print(f"Time for centerline '{centerline_name}': {int(minutes)} min {seconds:.2f} sec")
            time_log_file.write(f"Time for centerline '{centerline_name}': {int(minutes)} min {seconds:.2f} sec\n")

            count = count + 1
            print(count)

        overall_end = time.time()
        total_time = overall_end - overall_start
        print(f"Total simulation time: {total_time:.2f} seconds")
        minutes, seconds = divmod(total_time, 60)
        time_log_file.write(f"Total simulation time: {int(minutes)} min {seconds:.2f} sec\n")

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        


    finally:
        # Always save the final point cloud, even if interrupted
        final_path = os.path.join("pointclouds", "final_point_cloud.ply")
        time_log_file.close()
        overall_end = time.time()
        total_time = overall_end - overall_start
        print(f"Total simulation time: {total_time:.2f} seconds")
        minutes, seconds = divmod(total_time, 60)
        time_log_file.write(f"Total simulation time: {int(minutes)} min {seconds:.2f} sec\n")

        if len(point_cloud_generator.pcd.points) > 0:
            print(f"Total points in point cloud before saving: {len(point_cloud_generator.pcd.points)}")
            point_cloud_generator.save_accumulated_point_cloud()
            print(f"Final point cloud saved at {final_path}.")
        
        else:
            
            print("Point cloud is empty. Nothing to save.")