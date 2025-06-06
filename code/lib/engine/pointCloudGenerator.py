import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from collections import defaultdict
import cv2

class PointCloudGenerator:
    def __init__(self, intrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix
        self.pcd = o3d.geometry.PointCloud()
        self.reference_pcd = None  # Reference point cloud for alignment
        self.initial_transformation = np.identity(4)  # Store the last ICP transformation
        self.trimmed_accumulated_pcd = o3d.geometry.PointCloud()  # Initialize the trimmed accumulated point cloud

        # Clear the log file at the start of the program
        # Define the log directory relative to the script's location
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        log_dir = os.path.normpath(log_dir)  # Normalize the path for cross-platform compatibility
        os.makedirs(log_dir, exist_ok=True)

        ICP_log_file_path = os.path.join(log_dir, "ICP.txt")
        with open(ICP_log_file_path, "w") as ICP_log_file:
            ICP_log_file.write("Starting new ICP log set\n")

        last_selected_index = None  # Initialize last_selected_index
        last_selected_contour = None  # Initialize last_selected_contour

    def process_depth_and_transform(self, depth_img, transformation_matrix):
        """Process depth image and apply transformation."""
        """
        #Identify brightest pixel
        brightest_pixel = np.unravel_index(np.argmax(depth_img), depth_img.shape)
        print("Brightest pixel coordinates (y, x):", brightest_pixel)
        print("Brightest pixel depth value:", depth_img[brightest_pixel])

        #show depth image with brightest pixel
        depth_img_with_brightest = depth_img.copy()
        depth_img_with_brightest[brightest_pixel] = 1.0  # Set the brightest pixel to a different value for visibility

        # Identify groups of pixels with similar depth values
        # Threshold for depth similarity
        depth_threshold = 0.10  # 5 cm
        # Create a mask for pixels within the threshold
        depth_mask = np.abs(depth_img - depth_img[brightest_pixel]) < depth_threshold
        # Find contours of the masked regions
        contours, _ = cv2.findContours(depth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the original image
        depth_img_contours = cv2.cvtColor(depth_img_with_brightest, cv2.COLOR_GRAY2BGR)

        # Draw and number all contours for visualization
        depth_img_contours = cv2.cvtColor(depth_img_with_brightest, cv2.COLOR_GRAY2BGR)

        if len(contours) > 1:
            # Calculate centroids of all contours
            centroids = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))
                else:
                    centroids.append((0, 0))

            distance_threshold = 20  # pixels

            auto_select = False
            selected_idx = None

            if hasattr(self, "last_centroid") and self.last_centroid is not None:
                # If all centroids are close to the last centroid, auto-select previous
                if all(np.linalg.norm(np.array(self.last_centroid) - np.array(c)) < distance_threshold for c in centroids):
                    selected_idx = self.last_selected_index
                    auto_select = True
                    print(f"Contours are close. Automatically selecting previous index: {selected_idx}")
                else:
                    # Otherwise, select the closest if within threshold, else prompt
                    distances = [np.linalg.norm(np.array(self.last_centroid) - np.array(c)) for c in centroids]
                    min_idx = int(np.argmin(distances))
                    if distances[min_idx] < distance_threshold:
                        selected_idx = min_idx
                        print(f"Automatically selecting contour closest to previous centroid: {selected_idx}")
                    else:
                        # Draw and number all contours for visualization
                        depth_img_contours = cv2.cvtColor(depth_img_with_brightest, cv2.COLOR_GRAY2BGR)
                        for idx, contour in enumerate(contours):
                            cv2.drawContours(depth_img_contours, [contour], -1, (0, 255, 0), 2)
                            #cv2.imshow("Contours", depth_img_contours)
                            #cv2.waitKey(10000)

                            cX, cY = centroids[idx]
                            cv2.putText(depth_img_contours, str(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Contours (numbered)", depth_img_contours)
                        cv2.waitKey(10000)
                        selected_idx = int(input(f"Select contour index (0 to {len(contours)-1}): "))
            else:
                # First time: prompt user
                depth_img_contours = cv2.cvtColor(depth_img_with_brightest, cv2.COLOR_GRAY2BGR)
                for idx, contour in enumerate(contours):
                    cv2.drawContours(depth_img_contours, [contour], -1, (0, 255, 0), 2)
                    cX, cY = centroids[idx]
                    cv2.putText(depth_img_contours, str(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Contours (numbered)", depth_img_contours)
                cv2.waitKey(10000)
                selected_idx = int(input(f"Select contour index (0 to {len(contours)-1}): "))

            self.last_selected_index = selected_idx
            self.last_centroid = centroids[selected_idx]
            selected_contour = contours[selected_idx]
            self.last_selected_contour = selected_contour

            # Get the coordinates of the selected contour
            selected_contour_coords = selected_contour[:, 0, :]  # Reshape to (N, 2)
            # Calculate the centroid of the selected contour
            centroid = np.mean(selected_contour_coords, axis=0)
            print("Centroid of selected contour (x, y):", centroid)
            # Draw the centroid on the image
            depth_img_with_centroid = depth_img.copy()
            depth_img_with_centroid = cv2.cvtColor(depth_img_with_centroid, cv2.COLOR_GRAY2BGR)
            #cv2.imshow("Centroids", depth_img_with_centroid)
            #cv2.waitKey(10000)

            # Copy the original depth image
            depth_img_filled = depth_img.copy()

            # Create a mask for all contours except the selected one
            unselected_mask = np.zeros_like(depth_img, dtype=np.uint8)
            for idx, contour in enumerate(contours):
                if idx != selected_idx:
                    cv2.drawContours(unselected_mask, [contour], -1, 255, -1)  # Fill unselected contours white
            # Draw the selected contour in red for visibility
            cv2.drawContours(depth_img_filled, [selected_contour], -1, (0, 0, 255), 2)  # Draw selected contour in red
            #cv2.imshow("Selected Contour", depth_img_filled)
            #cv2.waitKey(10000)  # Display for 10 seconds

            # Fill all unselected contours with black in the depth image
            depth_img_filled = depth_img.copy()
            depth_img_filled[unselected_mask == 255] = 0

        else : 
            print("No contours found or only one contour detected.")
            depth_img_filled = depth_img.copy()
        """
        K = self.intrinsic_matrix
        height, width = depth_img.shape
        x, y = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
        x = (x - K[0, 2]) / K[0, 0]
        y = (y - K[1, 2]) / K[1, 1]

        # Apply a 180-degree rotation around the X-axis
        rotation_180_x = R.from_euler('x', 180, degrees=True).as_matrix()
        transformation_matrix[:3, :3] = transformation_matrix[:3, :3] @ rotation_180_x

        # Invert the Z translation (apply after rotation)
        transformation_matrix[2, 3] *= -1

        X = depth_img * x
        Y = depth_img * y
        Z = depth_img

        points_3d = np.vstack((X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())))
        #points_3d = np.vstack((Z.ravel(), Y.ravel(), X.ravel(), np.ones_like(X.ravel())))
        points_world = (transformation_matrix @ points_3d).T[:, :3]

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points_world)

        # Perform alignment using ICP
        if self.reference_pcd is not None:
            aligned_pcd = self.apply_icp(self.reference_pcd, temp_pcd, self.initial_transformation)
            if aligned_pcd is not None:
                temp_pcd = aligned_pcd
                # Update the reference point cloud to the latest aligned point cloud
                self.reference_pcd = temp_pcd
            else:
                print("\033[91mLow ICP fitness. Skipping transformation for this frame.\033[0m")
        else:
            print("No reference point cloud available. Skipping ICP alignment.")
            self.reference_pcd = temp_pcd # Set the first point cloud as the reference

        # Update the accumulated point cloud
        if not self.pcd.is_empty():
            self.pcd.points = o3d.utility.Vector3dVector(
                np.concatenate((np.asarray(self.pcd.points), np.asarray(temp_pcd.points)), axis=0)
            )
        else:
            self.pcd = temp_pcd

        # Debugging
        print(f"Accumulated point cloud size: {len(self.pcd.points)}")

        # Denoise and downsample the accumulated point cloud
        self.denoise_and_downsample()
        self.save_accumulated_point_cloud()


    def apply_icp(self, reference_pcd, target_pcd, initial_transformation=np.identity(4)):
        """Align the target point cloud to the reference using ICP."""

        # Save the ICP log to a file
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        ICP_log_file_path = os.path.join(log_dir, "ICP.txt")

        # Open the log file in append mode
        with open(ICP_log_file_path, "a+") as ICP_log_file:
            # Perform ICP alignment
            threshold = 0.089  # Distance threshold for ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                target_pcd, reference_pcd, threshold,
                initial_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            ICP_log_file.write(f"ICP Fitness: {icp_result.fitness}, Inlier RMSE: {icp_result.inlier_rmse}\n")

            # Skip transformation if fitness is too low
            if icp_result.fitness < 0.3:  # Adjust the threshold as needed
                ICP_log_file.write(f"ICP Fitness: {icp_result.fitness}, Low fitness. Skipping transformation.\n")
                print("\033[91mLow ICP fitness. Skipping transformation for this frame.\033[0m")
                return None  # Return None to indicate low fitness

            aligned_pcd = target_pcd.transform(icp_result.transformation)
            self.initial_transformation = icp_result.transformation  # Update the initial transformation for the next frame
            
            return aligned_pcd

    def denoise_and_downsample(self):
        """Apply denoising and downsampling to the point cloud."""
        if not self.pcd.is_empty():
            cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            self.pcd = self.pcd.select_by_index(ind)
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.008)
    
    def save_accumulated_point_cloud(self, pcd=None, filename="accumulated_point_cloud.npy"):
        """Save point cloud as a NumPy file in its own folder."""
        if pcd is None:
            pcd = self.pcd

            output_folder = os.path.join("pointclouds", "accumulated")
            os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

            # Ensure filename ends with .npy
            if not filename.endswith(".npy"):
                filename += ".npy"

            file_path = os.path.join(output_folder, filename)
            file_path = os.path.normpath(file_path)  # Normalize the path

            print(f"Saving to: {file_path}")  # Debug print

            # Check for invalid path (empty or forbidden characters)
            if not file_path or any(char in file_path for char in '<>:"|?*'):
                print(f"Invalid file path: {file_path}. Skipping save.")
                return

            np_points = np.asarray(self.pcd.points)
            try:
                np.save(file_path, np_points)
                print(f"Saved accumulated point cloud to {file_path}")
            except (OSError, ValueError) as e:
                print(f"Failed to save point cloud: {e}")
                return

        if pcd is None:
            print("Error: No valid point cloud to save.")
            return


    def save_intermediate_pointcloud(self, depth_img, transformation_matrix, filename):
        """Generate a new point cloud from a depth image, apply ICP, and save it."""
        """
        #Identify brightest pixel
        brightest_pixel = np.unravel_index(np.argmax(depth_img), depth_img.shape)
        print("Brightest pixel coordinates (y, x):", brightest_pixel)
        print("Brightest pixel depth value (m):", depth_img[brightest_pixel])

        #show depth image with brightest pixel
        depth_img_with_brightest = depth_img.copy()
        depth_img_with_brightest[brightest_pixel] = 1.0  # Set the brightest pixel to a different value for visibility

        # Identify groups of pixels with similar depth values
        # Threshold for depth similarity
        depth_threshold = 0.10  # 10 cm
        # Create a mask for pixels within the threshold
        depth_mask = np.abs(depth_img - depth_img[brightest_pixel]) < depth_threshold
        # Find contours of the masked regions
        contours, _ = cv2.findContours(depth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 1:

            # Prompt user to select a contour index
            selected_idx = self.last_selected_index
            selected_contour = contours[selected_idx]
            # Get the coordinates of the selected contour
            selected_contour_coords = selected_contour[:, 0, :]  # Reshape to (N, 2)
            # Calculate the centroid of the selected contour
            centroid = np.mean(selected_contour_coords, axis=0)
            print("Centroid of selected contour (x, y):", centroid)
           
            # Copy the original depth image
            depth_img_filled = depth_img.copy()

            # Create a mask for all contours except the selected one
            unselected_mask = np.zeros_like(depth_img, dtype=np.uint8)
            for idx, contour in enumerate(contours):
                if idx != selected_idx:
                    cv2.drawContours(unselected_mask, [contour], -1, 255, -1)  # Fill unselected contours white

            # Fill all unselected contours with black in the depth image
            depth_img_filled = depth_img.copy()
            depth_img_filled[unselected_mask == 255] = 0

        else : 
            print("No contours found or only one contour detected.")
            depth_img_filled = depth_img.copy()
        """
        
        K = self.intrinsic_matrix
        height, width = depth_img.shape
        x, y = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
        x = (x - K[0, 2]) / K[0, 0]
        y = (y - K[1, 2]) / K[1, 1]

        # Apply 180-degree rotation and Z-inversion
        rotation_180_x = R.from_euler('x', 180, degrees=True).as_matrix()
        transformation_matrix[:3, :3] = transformation_matrix[:3, :3] @ rotation_180_x
        #transformation_matrix[2, 3] *= -1

        X = depth_img * x
        Y = depth_img * y
        Z = depth_img

        points_3d = np.vstack((X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())))
        points_world = (transformation_matrix @ points_3d).T[:, :3]

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points_world)

        if temp_pcd.is_empty():
            print("Generated point cloud is empty. Skipping save.")
            return None

        # Save the (aligned or raw) point cloud
        output_folder = os.path.join("pointclouds", "intermediate_pointclouds")
        os.makedirs(output_folder, exist_ok=True)
        file_path_ply = os.path.normpath(os.path.join(output_folder, filename + ".ply"))

        # Denoise and downsample the intermediate point cloud
        self.denoise_and_downsample()

        try:
            o3d.io.write_point_cloud(file_path_ply, temp_pcd)
            print(f"Intermediate point cloud saved: {file_path_ply}")
        except OSError as e:
            print(f"Failed to save intermediate point cloud: {e}")
            return None

        return temp_pcd  # Aligned if possible
