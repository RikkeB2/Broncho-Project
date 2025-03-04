import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from lib.engine.onlineSimulationBO import onlineSimulationWithNetwork as onlineSimulatorBO
from lib.engine.pointCloudGenerator import PointCloudGenerator
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

    online_test_centerline_names_list = ['siliconmodel3 Centerline model'] + ['siliconmodel3 Centerline model_{}'.format(x) for x in range(1, 60)]
    #online_test_centerline_names_list = ['siliconmodel3BO Centerline model_{}'.format(x) for x in range(1, 20)]

    global_batch_count = 0
    count = 0

    # Ensure the pointclouds directory exists
    pointclouds_dir = "./pointclouds"
    if not os.path.exists(pointclouds_dir):
        os.makedirs(pointclouds_dir)

    # Initialize PointCloudGenerator
    intrinsic_matrix = np.array([[175 / 1.008, 0, 100],
                                 [0, 175 / 1.008, 100],
                                 [0, 0, 1]])
    point_cloud_generator = PointCloudGenerator(intrinsic_matrix)

    try:
        for online_test_centerline_name in online_test_centerline_names_list:
            simulator = onlineSimulatorBO(args.dataset_dir, online_test_centerline_name, renderer='pyrender', training=False)
            path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, originalCenterlineArray, safe_distance \
                = simulator.runManual(args, point_cloud_generator)

            count = count + 1
            print(count)

            # Save the point cloud periodically
            if count % 50 == 0:  # Adjust the frequency as needed
                print(f"Saving intermediate point cloud at step {count}")
                print(f"Total points in point cloud before saving: {len(point_cloud_generator.pcd.points)}")
                point_cloud_generator.save_pc(os.path.join(pointclouds_dir, f"intermediate_point_cloud_{count}.pcd"))
                print(f"Intermediate point cloud saved at step {count}")

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    finally:
        # Always save the final point cloud, even if interrupted
        final_path = os.path.join("pointclouds", "final_point_cloud.pcd")
        if len(point_cloud_generator.pcd.points) > 0:
            print(f"Total points in point cloud before saving: {len(point_cloud_generator.pcd.points)}")
            point_cloud_generator.save_pc(final_path)
            print(f"✅ Final point cloud saved at {final_path}.")
        else:
            print("❌ Point cloud is empty. Nothing to save.")