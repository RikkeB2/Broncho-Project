import numpy as np
import open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt


from ..lib.engine.centerLineGenerator import CenterlineGenerator

# === Load point cloud ===
pcd = o3d.io.read_point_cloud("code/pointclouds/intermediate_pointclouds/intermediate_point_cloud_20.ply")
pts = np.asarray(pcd.points)

# === Initialize generator and geodesic path ===
gen = CenterlineGenerator(num_points=100)
tree = cKDTree(pts)
d, nbrs = tree.query(pts, k=gen.k_neighbors + 1)
rows = np.repeat(np.arange(len(pts)), gen.k_neighbors)
cols = nbrs[:, 1:].ravel()
wts = d[:, 1:].ravel()
adj = coo_matrix((wts, (rows, cols)), shape=(len(pts), len(pts)))
adj = (adj + adj.T) * 0.5

centroid = pts.mean(axis=0)
A = np.argmax(np.linalg.norm(pts - centroid, axis=1))
B = np.argmax(np.linalg.norm(pts - pts[A], axis=1))
_, pred = dijkstra(adj, indices=A, return_predecessors=True)

path_idx = []
i = B
while i != A and i != -9999:
    path_idx.append(i)
    i = pred[i]
if i == A:
    path_idx.append(A)
path_pts = pts[path_idx[::-1]]  # Geodesic path

# === Interpolation ===
tck, u = splprep(path_pts.T, s=0)
u_fine = np.linspace(0, 1, 500)
interp_pts = np.array(splev(u_fine, tck)).T

# === Arc-length resample ===
tseg = np.linalg.norm(np.diff(path_pts, axis=0), axis=1)
cum = np.concatenate(([0], np.cumsum(tseg)))
tgt = np.linspace(0, cum[-1], gen.num_points)
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

# === LineSet helper for Open3D ===
def create_line_set(points, color):
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls

# === Open3D visualization ===
pcd.paint_uniform_color([0.7, 0.7, 0.7])
geo_ls = create_line_set(path_pts, [1, 0, 0])       # red
interp_ls = create_line_set(interp_pts, [0, 1, 0])  # green
resamp_ls = create_line_set(resampled, [0, 0, 1])   # blue

o3d.visualization.draw_geometries(
    [pcd, geo_ls, interp_ls, resamp_ls],
    window_name="3D Centerline Paths"
)

# === 2D Plot with Matplotlib ===
plt.figure(figsize=(10, 5))
plt.plot(path_pts[:, 1], path_pts[:, 0], 'r-', label='Geodesic Path (raw)')
plt.plot(interp_pts[:, 1], interp_pts[:, 0], 'g--', label='Spline Interpolation')
plt.plot(resampled[:, 1], resampled[:, 0], 'b.-', label='Arc-length Resampled')
plt.xlabel("Y")  # now on horizontal axis
plt.ylabel("X")  # now on vertical axis
plt.title("2D Projection (Y vs X)")
plt.axis('equal')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()