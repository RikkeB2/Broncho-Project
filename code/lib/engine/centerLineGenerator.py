import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point
from shapely.ops import polylabel
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import savgol_filter

class CenterlineGenerator:
    """
    Extracts the centerline of a tubular point cloud using an inscribed-circle
    method with clearance checks and smoothing.
    """
    def __init__(self, num_points=100, downsample_voxel=None, k_neighbors=8,
                 neigh_k=20, slab_factor=1.2, min_slice_count=30,
                 smooth_window=91, smooth_polyorder=3,
                 min_clearance=10.0, hull_area_thresh=5.0):
        """
        Initializes the CenterlineGenerator with user-defined parameters.

        Args:
            num_points (int): Number of centerline samples.
            downsample_voxel (float or None): Voxel size for downsampling, or None.
            k_neighbors (int): Connectivity for the geodesic graph.
            neigh_k (int): k-NN for slice thickness.
            slab_factor (float): Half-thickness multiplier for slicing pancake.
            min_slice_count (int): Minimum number of planar points to trust a slice.
            smooth_window (int): Window size for Savitzky–Golay smoothing. Must be odd.
            smooth_polyorder (int): Polynomial order for Savitzky–Golay smoothing. Smaller than smooth_window.
            min_clearance (float): Minimum clearance to wall in mm.
            hull_area_thresh (float): Minimum area to consider hull "well-formed" (mm^2).
        """
        self.num_points = num_points
        self.downsample_voxel = downsample_voxel
        self.k_neighbors = k_neighbors
        self.neigh_k = neigh_k
        self.slab_factor = slab_factor
        self.min_slice_count = min_slice_count
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder
        self.min_clearance = min_clearance
        self.hull_area_thresh = hull_area_thresh

    @staticmethod
    def _set_axes_equal(ax):
        """Utility: sets 3D plot axes to equal aspect ratio."""
        x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1]-x_limits[0]); y_range = abs(y_limits[1]-y_limits[0])
        z_range = abs(z_limits[1]-z_limits[0]); max_range = max(x_range,y_range,z_range)/2
        mid_x = np.mean(x_limits); mid_y = np.mean(y_limits); mid_z = np.mean(z_limits)
        ax.set_xlim3d(mid_x-max_range, mid_x+max_range)
        ax.set_ylim3d(mid_y-max_range, mid_y+max_range)
        ax.set_zlim3d(mid_z-max_range, mid_z+max_range)

    def extract_centerline_from_pcd(self, pcd: o3d.geometry.PointCloud):
        """
        Extracts the centerline from an open3d.geometry.PointCloud object using
        the inscribed-circle method.

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud.

        Returns:
            numpy.ndarray: The extracted centerline points, or None if extraction fails.
        """
        if pcd is None or pcd.is_empty():
            print("Warning: Received an empty point cloud. Centerline extraction skipped.")
            return None

        # Apply downsampling if specified
        if self.downsample_voxel:
            pcd = pcd.voxel_down_sample(self.downsample_voxel)

        pts = np.asarray(pcd.points)
        N = len(pts)

        if N < self.min_slice_count * 2:
            print(f"Warning: Point cloud has {N} points, which is less than the minimum required for robust centerline extraction ({self.min_slice_count * 2}). Skipping.")
            return None

        # 1) Build k-NN graph & Run Dijkstra for geodesic path
        try:
            tree = cKDTree(pts)
            d, nbrs = tree.query(pts, k=self.k_neighbors + 1)
            rows = np.repeat(np.arange(N), self.k_neighbors)
            cols = nbrs[:, 1:].ravel()
            wts = d[:, 1:].ravel()
            adj = coo_matrix((wts, (rows, cols)), shape=(N, N))
            adj = (adj + adj.T) * 0.5 # Ensure symmetric graph

            # Find approximate endpoints (farthest points from centroid)
            centroid = pts.mean(axis=0)
            A = np.argmax(np.linalg.norm(pts - centroid, axis=1))
            B = np.argmax(np.linalg.norm(pts - pts[A], axis=1))

            _, pred = dijkstra(adj, indices=A, return_predecessors=True)

            path_idx = []
            i = B
            while i != A and i != -9999: # Check both A and -9999 for termination
                path_idx.append(i)
                i = pred[i]
            if i == A: # Ensure start point is included if loop terminated there
                path_idx.append(A)
            else: # Path could not reach A, or B was unreachable
                print("Warning: Geodesic path could not connect endpoints. Returning None.")
                return None
            path_pts = pts[path_idx[::-1]] # Reverse to go A to B

        except Exception as e:
            print(f"Error during geodesic path finding: {e}")
            return None

        # 2) Arc-length resample
        try:
            tseg = np.linalg.norm(np.diff(path_pts, axis=0), axis=1)
            cum = np.concatenate(([0], np.cumsum(tseg)))
            # Ensure self.num_points is an integer, this is the most critical part from previous errors
            if not isinstance(self.num_points, int):
                raise TypeError(f"self.num_points must be an integer, but got {type(self.num_points)}: {self.num_points}")

            tgt = np.linspace(0, cum[-1], self.num_points)
            resampled = []
            for t in tgt:
                idx = np.searchsorted(cum, t)
                if idx == 0:
                    resampled.append(path_pts[0])
                elif idx == len(cum):
                    resampled.append(path_pts[-1])
                else:
                    t0, t1 = cum[idx - 1], cum[idx]
                    p0, p1 = path_pts[idx - 1], path_pts[idx]
                    a = (t - t0) / (t1 - t0)
                    resampled.append((1 - a) * p0 + a * p1)
            resampled = np.vstack(resampled)
        except Exception as e:
            print(f"Error during arc-length resampling: {e}")
            return None

        # 3) Compute Tangents
        try:
            tangs = np.zeros_like(resampled)
            for j in range(len(resampled)):
                if j == 0:
                    d = resampled[min(j + 2, len(resampled) - 1)] - resampled[j]
                elif j == len(resampled) - 1:
                    d = resampled[j] - resampled[max(0, j - 2)]
                else:
                    d = 0.5 * (resampled[min(j + 1, len(resampled) - 1)] - resampled[max(0, j - 1)])
                # Handle potential zero-norm for very close points
                norm_d = np.linalg.norm(d)
                tangs[j] = d / norm_d if norm_d > 1e-8 else np.array([0,0,1]) # Use a small default if norm is zero
        except Exception as e:
            print(f"Error during tangent computation: {e}")
            return None

        # 4) Inscribed-circle center with clearance check
        centers = []
        for idx, (cpt, tvec) in enumerate(zip(resampled, tangs)):
            half = self.slab_factor * np.mean(tree.query(cpt.reshape(1,-1), k=self.neigh_k)[0][0])
            mask = np.abs((pts - cpt).dot(tvec)) <= half
            slice_pts = pts[mask]

            if len(slice_pts) < self.min_slice_count:
                if 5 < idx < self.num_points - 5: # Only warn for interior slices
                    print(f"Warning: Slice {idx} has too few points ({len(slice_pts)}), using fallback center (geodesic point).")
                centers.append(cpt)
                continue

            # Project to local 2D plane
            vloc = slice_pts - cpt
            # Ensure tvec is normalized (it should be from previous step)
            tvec_norm = tvec / np.linalg.norm(tvec) if np.linalg.norm(tvec) > 1e-8 else np.array([0,0,1])

            # Choose an arbitrary orthogonal vector for the plane basis
            if np.abs(tvec_norm[0]) < 0.99: # If not mostly x-aligned
                v = np.array([1, 0, 0])
            else: # If mostly x-aligned, pick y-axis as arbitrary vector
                v = np.array([0, 1, 0])

            e1 = np.cross(tvec_norm, v)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(tvec_norm, e1)
            
            # Project points onto the 2D plane defined by e1 and e2
            xy = np.column_stack((vloc @ e1, vloc @ e2))
            
            # Create convex hull and find pole of inaccessibility
            try:
                hull = MultiPoint(xy).convex_hull
                if hull.is_empty or not hasattr(hull, 'exterior') or hull.area == 0:
                    print(f"Warning: Slice {idx} hull is empty, degenerate, or has zero area. Using fallback center (geodesic point).")
                    centers.append(cpt)
                    continue
                
                p = polylabel(hull, tolerance=0.01)
                dist_to_wall = hull.exterior.distance(p)
                # print(f"Slice {idx}: dist_to_wall={dist_to_wall:.2f} mm") # Consider removing for less verbose output

                if dist_to_wall < self.min_clearance:
                    if hull.area > self.hull_area_thresh: # Only apply aggressive adjustment for well-formed hulls
                        closest_point_on_hull = hull.exterior.interpolate(hull.exterior.project(p))
                        vec_to_wall = np.array([p.x - closest_point_on_hull.x, p.y - closest_point_on_hull.y])
                        norm_vec_to_wall = np.linalg.norm(vec_to_wall)

                        if norm_vec_to_wall > 1e-6: # Avoid division by zero
                            vec_to_wall_norm = vec_to_wall / norm_vec_to_wall
                            # Try to move 'p' inwards to meet MIN_CLEARANCE
                            new_xy = np.array([closest_point_on_hull.x, closest_point_on_hull.y]) + vec_to_wall_norm * self.min_clearance
                            if not hull.covers(Point(new_xy)):
                                # If the new point is outside the hull (meaning original pole is too close to a concave part),
                                # stick with the original pole, or perhaps just the closest point on the hull.
                                # For robustness, just use the original polylabel point here if it can't be moved in.
                                print(f"Warning: Could not enforce clearance at slice {idx}, adjusted point outside hull. Using original pole-of-inaccessibility.")
                                centers.append(cpt + p.x*e1 + p.y*e2)
                            else:
                                centers.append(cpt + new_xy[0]*e1 + new_xy[1]*e2)
                        else:
                            print(f"Warning: Could not compute inward direction at slice {idx}, using original pole-of-inaccessibility.")
                            centers.append(cpt + p.x*e1 + p.y*e2)
                    else:
                        # For small/degenerate hulls where clearance is violated, just use the original polylabel point
                        # without aggressive adjustment, as the hull shape might be unreliable.
                        centers.append(cpt + p.x*e1 + p.y*e2)
                        # print(f"Info: Slice {idx} hull area {hull.area:.2f} too small for strict clearance enforcement.") # Optional
                else:
                    # Clearance is sufficient, use the polylabel point directly
                    centers.append(cpt + p.x*e1 + p.y*e2)

                # Optional: Visualize problematic slices (every 10th only, if hull.area < 0.05 can be adjusted)
                # This visualization might slow down the processing, consider disabling for production.
                # if hull.area < 0.05 and idx % 10 == 0:
                #     plt.figure()
                #     plt.scatter(xy[:,0], xy[:,1], s=5, label='Slice points')
                #     if hasattr(hull, 'exterior'):
                #         hx, hy = hull.exterior.xy
                #         plt.plot(hx, hy, 'r-', label='Convex hull')
                #     # Plot the computed center for this slice
                #     final_cx, final_cy = centers[-1][0] - cpt[0], centers[-1][1] - cpt[1]
                #     plt.scatter(final_cx, final_cy, color='lime', marker='x', s=100, label='Final Center')
                #     plt.title(f'Slice {idx} - hull area: {hull.area:.4f}, clearance: {dist_to_wall:.2f}')
                #     plt.legend()
                #     plt.show()

            except Exception as e:
                print(f"Error processing slice {idx} for inscribed-circle: {e}. Using fallback center (geodesic point).")
                centers.append(cpt) # Fallback if any error occurs in shapely/clearance logic

        centerline_raw = np.vstack(centers)

        # 5) Smooth
        try:
            offs = centerline_raw - resampled
            n = offs.shape[0]
            # Adjust window size to be odd and not exceed n
            win = self.smooth_window
            if not (win <= n and win % 2 == 1):
                win = n if n % 2 else n - 1 # Ensure odd and max possible if too large/even
                if win < 3: win = 3 # Minimum window size
            
            poly = min(self.smooth_polyorder, win - 1) # Polynomial order must be less than window size

            sm = np.zeros_like(offs)
            for d in range(3):
                sm[:, d] = savgol_filter(offs[:, d], win, poly, mode='mirror') # Use 'mirror' mode for ends
            centerline = resampled + sm
            
        except Exception as e:
            print(f"Error during smoothing: {e}")
            return None

        return centerline

    def save_centerline(self, centerline: np.ndarray, filename="centerline.npy", output_dir="generated_centerlines/centerlines"):
        """
        Saves the extracted centerline to a .npy file.

        Args:
            centerline (numpy.ndarray): The extracted centerline points.
            filename (str): Name of the file to save the centerline (.npy).
            output_dir (str): Directory where the centerline will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        if not filename.endswith(".npy"):
            filename += ".npy"

        file_path = os.path.join(output_dir, filename)
        file_path = os.path.normpath(file_path)

        # Basic check for invalid path characters (more robust validation might be needed for production)
        if not file_path or any(char in file_path for char in '<>:"|?*'):
            print(f"Invalid file path: {file_path}. Skipping save.")
            return

        try:
            np.save(file_path, centerline)
            print(f"Centerline saved to {file_path}")
        except (OSError, ValueError) as e:
            print(f"Failed to save centerline: {e}")
