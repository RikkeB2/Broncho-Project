import numpy as np
import open3d as o3d
import os

def ply_to_npy(ply_path, npy_path=None):
    pcd = o3d.io.read_point_cloud(ply_path)
    npy_path = npy_path or os.path.splitext(ply_path)[0] + ".npy"
    np.save(npy_path, np.asarray(pcd.points))
    print(f"Saved: {npy_path}")

def npy_to_ply(npy_path, ply_path=None):
    points = np.load(npy_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ply_path = ply_path or os.path.splitext(npy_path)[0] + ".ply"
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved: {ply_path}")

def ply_to_obj(ply_path, obj_path=None):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    obj_path = obj_path or os.path.splitext(ply_path)[0] + ".obj"
    if not mesh.has_triangles():
        print("No mesh found, converting point cloud instead.")
        pcd = o3d.io.read_point_cloud(ply_path)
        o3d.io.write_triangle_mesh(obj_path, pcd)
    else:
        o3d.io.write_triangle_mesh(obj_path, mesh)
    print(f"Saved: {obj_path}")

def npy_to_obj(npy_path, obj_path=None):
    points = np.load(npy_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obj_path = obj_path or os.path.splitext(npy_path)[0] + ".obj"
    o3d.io.write_triangle_mesh(obj_path, pcd)
    print(f"Saved: {obj_path}")

if __name__ == "__main__":
    # Example usage
    # ply_to_npy("example.ply")
    # npy_to_ply("example.npy")
    # ply_to_obj("example.ply")
    # npy_to_obj("example.npy")
    pass