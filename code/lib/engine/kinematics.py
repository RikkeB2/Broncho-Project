import numpy as np

class Kinem:
    def __init__(self, r_cable=2.0, n_links=10):
        """
        r_cable: radius of cable ring (mm)
        n_links: number of discrete links in the constant-curvature model
        """
        self.r_cable = r_cable
        self.n_links = n_links

    @staticmethod
    def load_ply_xyz(path):
        """
        Load an ASCII PLY file and return an (N×3) numpy array of vertex positions.
        """
        pts = []
        with open(path, 'r') as f:
            line = f.readline().strip()
            if not line.startswith('ply'):
                raise ValueError('Not a PLY file')
            while True:
                line = f.readline().strip()
                if line.lower() == 'end_header':
                    break
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                x, y, z = map(float, parts[:3])
                pts.append([x, y, z])
        if not pts:
            raise ValueError('No vertex data found in PLY')
        return np.array(pts)

    def bronchoscope_forward_kinematics(self, T0, yaw, bend, insertion):
        """
        Constant-curvature FK:
        - insertion along local Z
        - bend about local Y by 'bend'
        - yaw about local Z by 'yaw'
        T0: 4×4 base frame, returns 4×4 tip frame
        """
        # insertion translation
        T_ins = np.eye(4)
        T_ins[2, 3] = insertion
        # bend about Y-axis
        cb, sb = np.cos(bend), np.sin(bend)
        T_bend = np.array([
            [ cb, 0, sb, 0],
            [  0, 1,  0, 0],
            [-sb, 0, cb, 0],
            [  0, 0,  0, 1]
        ])
        # yaw about Z-axis
        cy, sy = np.cos(yaw), np.sin(yaw)
        T_yaw = np.array([
            [cy, -sy, 0, 0],
            [sy,  cy, 0, 0],
            [ 0,   0, 1, 0],
            [ 0,   0, 0, 1]
        ])
        return T0 @ T_ins @ T_bend @ T_yaw

    def inverse_kinematics(self, T_des):
        """
        Given a desired frame T_des (4×4), extract insertion, bend, and yaw
        that align the scope's local axes to T_des.

        Returns:
            d (insertion, mm), phi (bend, rad), theta (yaw, rad)
        """
        R_des = T_des[:3, :3]
        # yaw from x-axis projection in XY
        theta = np.arctan2(R_des[1, 0], R_des[0, 0])
        # bend from Z-axis tilt
        phi = np.arccos(np.clip(R_des[2, 2], -1.0, 1.0))
        # insertion at frame origin
        d = 0.0
        return d, phi, theta

    def inverse_kinematics_path(self, centerline_points):
        """
        Build Frenet-like frames along the centerline and solve IK per point.

        centerline_points: N×3 array of XYZ samples.

        Returns:
            poses: list of 4×4 tip transforms (orientation only, origins on centerline)
            thetas: list of yaw angles (rad)
            phis: list of bend angles (rad)
            ds: list of insertion distances (mm)
        """
        pts = np.asarray(centerline_points)
        N = len(pts)
        # calculate tangents
        tangents = np.zeros_like(pts)
        tangents[0] = pts[1] - pts[0]
        for i in range(1, N-1):
            tangents[i] = pts[i+1] - pts[i-1]
        tangents[-1] = pts[-1] - pts[-2]
        # normalize
        lengths = np.linalg.norm(tangents, axis=1)
        tangents /= lengths[:, None]
        # cumulative insertion (arc length)
        ds = np.insert(np.cumsum(lengths[:-1]), 0, 0.0)

        poses, thetas, phis, ds_list = [], [], [], []
        world_up = np.array([0.0, 0.0, 1.0])

        for p, t, d in zip(pts, tangents, ds):
            # build frame
            z = t
            x = np.cross(world_up, z)
            if np.linalg.norm(x) < 1e-6:
                x = np.cross(np.array([1.0, 0.0, 0.0]), z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            F = np.eye(4)
            F[:3, 0] = x
            F[:3, 1] = y
            F[:3, 2] = z
            F[:3, 3] = p
            # extract IK
            d_val, phi, theta = self.inverse_kinematics(F)
            # but use actual insertion
            ds_list.append(d)
            phis.append(phi)
            thetas.append(theta)
            # local orientation
            cb, sb = np.cos(phi), np.sin(phi)
            T_bend = np.array([[cb,0,sb,0],[0,1,0,0],[-sb,0,cb,0],[0,0,0,1]])
            cy, sy = np.cos(theta), np.sin(theta)
            T_yaw = np.array([[cy,-sy,0,0],[sy,cy,0,0],[0,0,1,0],[0,0,0,1]])
            T_orient = T_bend @ T_yaw
            # place at p
            T_global = F @ T_orient
            poses.append(T_global)
        return poses, thetas, phis, ds_list