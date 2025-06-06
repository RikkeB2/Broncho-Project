from ctypes import windll
from re import X
from turtle import left
import os
from cv2 import TM_CCOEFF_NORMED
from graphviz import render
import pybullet as p
import pybullet_data
from mayavi import mlab
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import cv2
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from pyrender import IntrinsicsCamera, PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
import open3d as o3d
import threading
from scipy.spatial.transform import Rotation as RR

from .camera import fixedCamera
from .keyBoardEvents import getDirectionBO, getAdditionBO

from scipy.io import savemat

from .simRobot_kin import BroncoRobot2

from .pointCloudGenerator import PointCloudGenerator
from .centerLineGenerator import CenterlineGenerator
from .kinematics import Kinem
from scipy.spatial.transform import Rotation as R

def print_diagnostic(target_pose, actual_pose, label=""):
    R_target = target_pose[:3, :3]
    R_actual = actual_pose[:3, :3]

    # Compute rotation difference
    R_diff = R_target @ R_actual.T
    rot_diff = R.from_matrix(R_diff)
    euler_diff_deg = rot_diff.as_euler('xyz', degrees=True)
    angle_diff_deg = np.linalg.norm(euler_diff_deg)

    # Quaternions
    q_target = R.from_matrix(R_target).as_quat()
    q_actual = R.from_matrix(R_actual).as_quat()

    # Euler angles
    euler_target = R.from_matrix(R_target).as_euler('xyz', degrees=True)
    euler_actual = R.from_matrix(R_actual).as_euler('xyz', degrees=True)

    print(f"\n--- DIAGNOSTIC: Pose Comparison {label} ---")
    print("Target Euler Angles (deg):", euler_target)
    print("Actual Euler Angles (deg):", euler_actual)
    print("Rotation Difference (Euler xyz, deg):", euler_diff_deg)
    print("Rotation Difference Magnitude (deg):", angle_diff_deg)
    print("Target Quaternion [x, y, z, w]:", q_target)
    print("Actual Quaternion [x, y, z, w]:", q_actual)
    print("Translation Target:", target_pose[:3, 3])
    print("Translation Actual:", actual_pose[:3, 3])
    print("Translation Difference:", target_pose[:3, 3] - actual_pose[:3, 3])


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def apply_control_pad_icon(image, direction):
    # color = (0, 165, 255)
    color = (204, 0, 51)
    offset = np.array([-160, -50])
    up_arrow = np.array([[255, 90], [240, 105], [270, 105]]) + offset
    down_arrow = np.array([[255, 170], [240, 155], [270, 155]]) + offset
    left_arrow = np.array([[210, 130], [225, 115], [225, 145]]) + offset
    right_arrow = np.array([[300, 130], [285, 115], [285, 145]]) + offset
    # front_arrow = np.array([[255, 125], [245, 135], [265, 135]])
    front_rect = np.array([[245, 120], [265, 140]]) + offset
    cv2.drawContours(image, [up_arrow], 0, color, 2)
    cv2.drawContours(image, [down_arrow], 0, color, 2)
    cv2.drawContours(image, [left_arrow], 0, color, 2)
    cv2.drawContours(image, [right_arrow], 0, color, 2)
    # cv2.drawContours(image, [front_arrow], 0, (255, 255, 0), 2)
    # cv2.circle(image, [255, 130], 10, (255, 255, 0), 2)
    cv2.rectangle(image, front_rect[0], front_rect[1], color, 2)
    if direction == [1, 0, 0, 0, 0]:
        # cv2.putText(image, 'Up', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [up_arrow], 0, color, -1)
    elif direction == [0, 1, 0, 0, 0]:
        # cv2.putText(image, 'Left', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [left_arrow], 0, color, -1)
    elif direction == [0, 0, 1, 0, 0]:
        # cv2.putText(image, 'Down', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [down_arrow], 0, color, -1)
    elif direction == [0, 0, 0, 1, 0]:
        # cv2.putText(image, 'Right', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [right_arrow], 0, color, -1)
    elif direction == [0, 0, 0, 0, 1]:
        # cv2.putText(image, 'Straight', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # cv2.drawContours(image, [front_arrow], 0, (255, 255, 0), -1)
        # cv2.circle(image, [255, 130], 10, (255, 255, 0), 2)
        cv2.rectangle(image, front_rect[0], front_rect[1], color, -1)
    else:
        raise NotImplementedError()

    return image


def dcm2quat(R):
	
    epsilon = 1e-5
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    assert trace > -1
    if np.fabs(trace + 1) < epsilon:
        if np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 0:
            t = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / t
            q1 = t / 4
            q2 = (R[0, 2] + R[2, 0]) / t
            q3 = (R[0, 1] + R[1, 0]) / t
        elif np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 1:
            t = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / t
            q1 = (R[0, 1] + R[1, 0]) / t
            q2 = t / 4
            q3 = (R[2, 1] + R[1, 2]) / t
        else:
            t = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            q0 = (R[1, 0] - R[0, 1]) / t
            q1 = (R[0, 2] + R[2, 0]) / t
            q2 = (R[1, 2] - R[2, 1]) / t
            q3 = t / 4
    else:
        q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

    return np.array([q1, q2, q3, q0])

class onlineSimulationWithNetwork(object):

    def __init__(self, dataset_dir, centerline_name, renderer=None, training=True):

        # 1 - vai correr aqui para cada centerline!!

        # Load models
        name = centerline_name.split(" ")[0]
        self.bronchus_model_dir = os.path.join("airways", "AirwayHollow_{}_simUV.obj".format(name)) #parte externa
        self.airway_model_dir = os.path.join("airways", "AirwayModel_Peach_{}.vtk".format(name)) #parte interna, ar
        self.centerline_name = centerline_name
        centerline_model_name = centerline_name.lstrip(name + " ")
        self.centerline_model_dir = os.path.join("airways", "centerline_models_{}".format(name), centerline_model_name + ".obj")

        p.connect(p.GUI) #abre o pybullet para simulacao
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1. / 120.) #120Hz a velocidade da simulacao
        # useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
        p.loadURDF("plane100.urdf", useMaximalCoordinates=True) # modelo que ja tem no local de instalacao.

        shift = [0, 0, 0]
        meshScale = [0.01, 0.01, 0.01]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName=self.bronchus_model_dir,
                                            rgbaColor=[249 / 255, 204 / 255, 226 / 255, 1],
                                            specularColor=[0, 0, 0],
                                            visualFramePosition=shift,
                                            meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=self.bronchus_model_dir,
                                                collisionFramePosition=shift,
                                                meshScale=meshScale)

        # Augment on roll angle
        self.rand_roll = 0
        
        euler = p.getEulerFromQuaternion([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]) # Euler angles given as XYZ (pitch;roll;yaw)
        self.quaternion_model = p.getQuaternionFromEuler([np.pi / 2, self.rand_roll, 0])  # 4 floating point values [X,Y,Z,W]
        self.matrix_model = p.getMatrixFromQuaternion(self.quaternion_model)
        self.R_model = np.reshape(self.matrix_model, (3, 3))
        self.t_model = np.array([0, 0, 5])

        airwayBodyId = p.createMultiBody(baseMass=1,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0, 0, 5],
                                            baseOrientation=self.quaternion_model,
                                            useMaximalCoordinates=True)

        #p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1) #HERE

        # Set camera path (centerline defined)
        file_path = self.centerline_model_dir
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_path)
        reader.Update()
        self.vtkdata = reader.GetOutput()

        # Collision detection
        self.pointLocator = vtk.vtkPointLocator()
        self.pointLocator.SetDataSet(self.vtkdata)
        self.pointLocator.BuildLocator()

        self.camera = fixedCamera(0.01, p)
        #self.camera.width = 800
        #self.camera.height = 800

        boundingbox = p.getAABB(airwayBodyId)
        print(boundingbox)
        #print(np.max(centerlineArray, axis=0))
        #print(np.min(centerlineArray, axis=0))
        #print(np.argmax(centerlineArray, axis=0))
        position = p.getBasePositionAndOrientation(airwayBodyId) # position XYZ e orientacao quaternion

        # Pyrender initialization
        self.renderer = renderer #pyrender
        fuze_trimesh = trimesh.load(self.bronchus_model_dir)

        fuze_mesh = Mesh.from_trimesh(fuze_trimesh)
        spot_l = SpotLight(color=np.ones(3), intensity=0.3,
                        innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        self.cam = IntrinsicsCamera(fx=175 / 1.008, fy=175 / 1.008, cx=200, cy=200, znear=0.00001)
        self.scene = Scene(bg_color=(0., 0., 0.))
        self.fuze_node = Node(mesh=fuze_mesh, scale=meshScale, rotation=self.quaternion_model, translation=self.t_model)
        self.scene.add_node(self.fuze_node)
        self.spot_l_node = self.scene.add(spot_l)
        self.cam_node = self.scene.add(self.cam)
        self.r = OffscreenRenderer(viewport_width=400, viewport_height=400)

        self.update_count = 0


    def indexFromDistance(self, centerlineArray, count, distance):
        centerline_size = len(centerlineArray)
        start_index = count
        cur_index = start_index
        centerline_length = 0
        if cur_index <= 0:
            return False
        while(1):
            length_diff = np.linalg.norm(centerlineArray[cur_index - 1] - centerlineArray[cur_index]) # anda para tras porque os dados estao ao contrario
            centerline_length += length_diff
            cur_index -= 1
            if cur_index <= 0:
                return False
            if centerline_length > distance:
                return cur_index

    
    def get_images(self, yaw, pitch, t, pos_vector):
        rgb_img_bullet, _, _ = self.camera.lookat(yaw, pitch, t, -pos_vector) # for visulization (a camara esta rodada em relacao ao mundo Z=Y, Y = -Z)
        rgb_img_bullet = rgb_img_bullet[:, :, :3]
        rgb_img_bullet = cv2.resize(rgb_img_bullet, (200, 200))
        rgb_img_bullet = np.transpose(rgb_img_bullet, axes=(2, 0, 1))
        pitch = pitch / 180 * np.pi + np.pi / 2
        yaw = yaw / 180 * np.pi
        quat = p.getQuaternionFromEuler([pitch, 0, yaw])
        R = p.getMatrixFromQuaternion(quat)
        R = np.reshape(R, (3, 3))
        pose = np.identity(4)
        pose[:3, 3] = t
        pose[:3, :3] = R
        light_intensity = 0.3
        self.scene.clear()
        self.scene.add_node(self.fuze_node)
        spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
            innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        spot_l_node = self.scene.add(spot_l, pose=pose)
        cam_node = self.scene.add(self.cam, pose=pose)
        self.scene.set_pose(spot_l_node, pose)
        self.scene.set_pose(cam_node, pose)
        rgb_img, depth_img = self.r.render(self.scene)
        rgb_img_ori = rgb_img.copy()
        rgb_img = rgb_img[:, :, :3]

        mean_intensity = np.mean(rgb_img)
        count_AE = 0
        min_light_intensity = 0.001
        max_light_intensity = 20
        while np.abs(mean_intensity - 140) > 20:
            if count_AE > 1000:
                break
            if np.abs(min_light_intensity - light_intensity) < 1e-5 or np.abs(max_light_intensity - light_intensity) < 1e-5:
                break
            if mean_intensity > 140:
                max_light_intensity = light_intensity
                light_intensity = (min_light_intensity + max_light_intensity) / 2
            else:
                min_light_intensity = light_intensity
                light_intensity = (min_light_intensity + max_light_intensity) / 2
            self.scene.clear()
            self.scene.add_node(self.fuze_node)
            spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                    innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            spot_l_node = self.scene.add(spot_l, pose=pose)
            cam_node = self.scene.add(self.cam, pose=pose)
            self.scene.set_pose(spot_l_node, pose)
            self.scene.set_pose(cam_node, pose)
            rgb_img, depth_img = self.r.render(self.scene)
            rgb_img_ori = rgb_img.copy()
            rgb_img = rgb_img[:, :, :3]
            mean_intensity = np.mean(rgb_img)
            count_AE += 1

        mean_intensity = print("Mean intensity:", np.mean(rgb_img))

        rgb_img = cv2.resize(rgb_img, (200, 200))
        rgb_img = np.transpose(rgb_img, axes=(2, 0, 1))
        if self.renderer == 'pybullet':
            rgb_img = rgb_img_bullet

        depth_img = depth_img * 255
        depth_img = depth_img.astype(np.uint8)
        depth_img = cv2.resize(depth_img, (200, 200))

        return rgb_img, depth_img, rgb_img_ori


    def get_imagesPRY(self, yaw, pitch, roll, t, pos_vector):
        rgb_img_bullet, _, _ = self.camera.lookatBO(yaw, pitch, roll, t, -pos_vector) # for visulization (a camara esta rodada em relacao ao mundo Z=Y, Y = -Z)
        #if isinstance(rgb_img_bullet, tuple):
            #rgb_img_bullet = np.array(rgb_img_bullet)
        rgb_img_bullet = rgb_img_bullet[:, :, :3]
        rgb_img_bullet = cv2.resize(rgb_img_bullet, (200, 200))
        rgb_img_bullet = np.transpose(rgb_img_bullet, axes=(2, 0, 1))
        pitch = pitch / 180 * np.pi + np.pi / 2
        yaw = yaw / 180 * np.pi
        roll = roll / 180 * np.pi

        quat = p.getQuaternionFromEuler([pitch, roll, yaw])
        R = p.getMatrixFromQuaternion(quat)
        R = np.reshape(R, (3, 3))
        #Use this pose
        pose = np.identity(4)
        pose[:3, 3] = t
        pose[:3, :3] = R
        light_intensity = 0.3
        self.scene.clear()
        self.scene.add_node(self.fuze_node)
        spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
            innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        spot_l_node = self.scene.add(spot_l, pose=pose)
        cam_node = self.scene.add(self.cam, pose=pose)
        self.scene.set_pose(spot_l_node, pose)
        self.scene.set_pose(cam_node, pose)
        rgb_img, depth_img = self.r.render(self.scene)
        rgb_img_ori = rgb_img.copy()
        rgb_img = rgb_img[:, :, :3]

        mean_intensity = np.mean(rgb_img)
        count_AE = 0
        min_light_intensity = 0.001
        max_light_intensity = 20
        while np.abs(mean_intensity - 140) > 20:
            if count_AE > 1000:
                break
            if np.abs(min_light_intensity - light_intensity) < 1e-5 or np.abs(max_light_intensity - light_intensity) < 1e-5:
                break
            if mean_intensity > 140:
                max_light_intensity = light_intensity
                light_intensity = (min_light_intensity + max_light_intensity) / 2
            else:
                min_light_intensity = light_intensity
                light_intensity = (min_light_intensity + max_light_intensity) / 2
            self.scene.clear()
            self.scene.add_node(self.fuze_node)
            spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                    innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            spot_l_node = self.scene.add(spot_l, pose=pose)
            cam_node = self.scene.add(self.cam, pose=pose)
            self.scene.set_pose(spot_l_node, pose)
            self.scene.set_pose(cam_node, pose)
            rgb_img, depth_img = self.r.render(self.scene)
            rgb_img_ori = rgb_img.copy()
            rgb_img = rgb_img[:, :, :3]
            mean_intensity = np.mean(rgb_img)
            count_AE += 1

        mean_intensity = print("Mean intensity:", np.mean(rgb_img))

        rgb_img = cv2.resize(rgb_img, (200, 200))
        rgb_img = np.transpose(rgb_img, axes=(2, 0, 1))
        if self.renderer == 'pybullet':
            rgb_img = rgb_img_bullet

        depth_img2 = depth_img.copy()
        
        depth_img3 = depth_img.copy()
        depth_img3 = np.asarray(depth_img3, dtype=np.float32)

        # Debugging: Print depth image stats before normalization
        print(f"Depth Image Stats before normalization: Min={np.min(depth_img)}, Max={np.max(depth_img)}, Mean={np.mean(depth_img)}")

        depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img)) 
        depth_img = depth_img * 255
        depth_img = depth_img.astype(np.uint8)
        depth_img = cv2.resize(depth_img, (200, 200))

        # Debugging: Print depth image stats after normalization
        print(f"Depth Image Stats after normalization: Min={np.min(depth_img)}, Max={np.max(depth_img)}, Mean={np.mean(depth_img)}")

        return pose, rgb_img, depth_img, rgb_img_ori, depth_img2, depth_img3
    
    def get_transformation_matrix(self, R, T):
        """Construct the transformation matrix from the rotation matrix and translation vector."""
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = T

        return transformation_matrix
    
    def load_centerline(self, centerline_name):
        # Only update centerline-related attributes, do not re-initialize PyBullet or the scene!
        name = centerline_name.split(" ")[0]
        centerline_model_name = centerline_name.lstrip(name + " ")
        self.centerline_model_dir = os.path.join("airways", "centerline_models_{}".format(name), centerline_model_name + ".obj")

        # Load and process the new centerline
        reader = vtk.vtkOBJReader()
        reader.SetFileName(self.centerline_model_dir)
        reader.Update()
        mesh = reader.GetOutput()
        points = mesh.GetPoints()
        data = points.GetData()
        centerlineArray = vtk_to_numpy(data)
        centerlineArray = np.dot(self.R_model, centerlineArray.T).T * 0.01 + self.t_model

        # Downsample/upsample and smooth as in your __init__ (copy that logic here)
        centerline_length = 0
        for i in range(len(centerlineArray) - 1):
            length_diff = np.linalg.norm(centerlineArray[i] - centerlineArray[i + 1])
            centerline_length += length_diff
        centerline_size = len(centerlineArray)
        lenth_size_rate = 0.007
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

        # Smoothing
        centerlineArray_smoothed = np.zeros_like(centerlineArray)
        for i in range(len(centerlineArray)):
            left_bound = i - 10
            right_bound = i + 10
            if left_bound < 0: left_bound = 0
            if right_bound > len(centerlineArray): right_bound = len(centerlineArray)
            centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound : right_bound], axis=0)
        self.centerlineArray = centerlineArray_smoothed
        self.originalCenterlineArray = centerlineArray
    
    def runVS2(self, args, point_cloud_generator, centerline_generator):

        #Inverse the centerline array to match the simulation
        self.centerlineArray = np.flip(self.centerlineArray, axis=0)
        # Compute total centerline length
        self.centerline_length = np.sum(
            np.linalg.norm(np.diff(self.centerlineArray, axis=0), axis=1)
        )

        # Remove previous debug lines if any
        if hasattr(self, "debug_line_ids"):
            for line_id in self.debug_line_ids:
                p.removeUserDebugItem(line_id)
        self.debug_line_ids = []

        # Draw only the current centerline
        for i in range(len(self.centerlineArray) - 1):
            line_id = p.addUserDebugLine(
                self.centerlineArray[i], self.centerlineArray[i + 1],
                lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3
            )
            self.debug_line_ids.append(line_id)

        # Pitch and Roll
        count = len(self.centerlineArray) - 1
        start_index = len(self.centerlineArray) - 3
        x, y, z = self.centerlineArray[start_index]
        yaw = 0
        pitch = 0
        roll = 0

        ######################
        # Defining initial camera orientation to always aligned with the path
        pos_vector_gt = (self.centerlineArray[count - 1] - self.centerlineArray[count]) / np.linalg.norm(self.centerlineArray[count - 1] - self.centerlineArray[count]) # Ideal motion direction
        pos_vector_norm = np.linalg.norm(pos_vector_gt)

        pitch = np.arcsin(pos_vector_gt[2] / pos_vector_norm)
        if pos_vector_gt[0] > 0:
            yaw = -np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))
        else:
            yaw = np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))
        ##############################

        quat_init = p.getQuaternionFromEuler([pitch, roll, yaw])
        R_init = p.getMatrixFromQuaternion(quat_init)
        R_init = np.reshape(R_init, (3, 3))
        t_init = np.array([x, y, z])

        print("Initial position:", x, y, z)
        print("Initial orientation:", pitch, roll, yaw)


        # Build initial transform by converting Euler angles to quaternion to rotation matrix
        quat_init = p.getQuaternionFromEuler([pitch, roll, yaw])
        R_init = p.getMatrixFromQuaternion(quat_init)
        R_init = np.reshape(R_init, (3, 3))
        t_init = np.array([x, y, z])
        T_current = t_init.copy()
        R_current = R_init.copy()
        pos_vector_init = self.centerlineArray[count - 1] - self.centerlineArray[count]

        # Debug visualisation of centerline (draws green line segments along entire centerline for refernce)
        for i in range(len(self.centerlineArray) - 1):
            p.addUserDebugLine(self.centerlineArray[i], self.centerlineArray[i + 1], lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3)
            
        #for i in range(len(self.centerlineArray2) - 1):
        #    p.addUserDebugLine(self.centerlineArray2[i], self.centerlineArray2[i + 1], lineColorRGB=[0, 0, 1], lifeTime=0, lineWidth=5)

        path_length = 0
        path_centerline_ratio_list = []
        safe_distance_list = []

        path_trajectoryT = []
        path_trajectoryR = []

        path_joint = []
        path_jointvel = []

        ###
        m_distBO = 0.003 # distancia a percorrer em cada iteracao mm

        args.human = True # Keep it True for now, but the movement will be IK-driven
        direction = np.array([0, 0, 0])

        # Matrix for simulated camera, scaled for resolution.

        intrinsic_matrix = np.array([[181.93750381, 0, 200],
                            [0, 181.93750381, 200],
                            [0, 0, 1]])  #HERE BO

        target_width = 200
        target_height = 200

        original_width = 500 #estimated
        original_height = 500 #estimated

        scale_x = target_width / original_width
        scale_y = target_height / original_height

        #Intantiate bronchoscope robot controller (BroncoRobot1) - Assuming this is defined elsewhere
        # m_robot = BroncoRobot1()

        # Opens text files to record every transformation, pose, translation, rotation, and kinematic data
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create log files in the log directory
        transformation_log_file = open(os.path.join(log_dir, "transformation.txt"), "w+")
        pose_log_file = open(os.path.join(log_dir, "pose.txt"), "w+")
        translation_log_file = open(os.path.join(log_dir, "translation.txt"), "w+")
        translation_difference_log_file = open(os.path.join(log_dir, "translation_difference.txt"), "w+")
        rotation_log_file = open(os.path.join(log_dir, "rotation.txt"), "w+")
        rotation_log_file_2 = open(os.path.join(log_dir, "rotation_2.txt"), "w+")
        kinematic_log_file = open(os.path.join(log_dir, "kinematic.txt"), "w+")
        kinematic_deviation_log_file = open(os.path.join(log_dir, "kinematic_deviation.txt"), "w+")
        centerline_traversed_log_file = open(os.path.join(log_dir, "centerline_traversed.txt"), "w+")
        # log joint data
        joint_log_file = open(os.path.join("./logs", "joint_data.txt"), "w+")
        # log joint data (forward, backward and bending)
        joint_log_file_forward = open(os.path.join("./logs", "joint_data_forward.txt"), "w+")
        joint_log_file_backward = open(os.path.join("./logs", "joint_data_backward.txt"), "w+")
        collision_log_file = open(os.path.join(log_dir, "collision.txt"), "w+") 
        cartesian_log_file = open(os.path.join(log_dir, "cartesian_data.txt"), "w+")
        centerline_euclidean_distance_log_file = open(os.path.join(log_dir, "centerline_euclidean_distance.txt"), "w+")


        frame_count = 0

        accumulated_points = None
        accumulated_lines = []
        forward_orientations = []

        # Initialize the Kinem class before the loop
        kinem = Kinem(r_cable=2.0, n_links=10)
        num_points = len(self.centerlineArray)
        current_centerline_index = 0


        # Calculate the IK path once at the beginning
        poses, thetas, phis, ds = kinem.inverse_kinematics_path(self.centerlineArray)
        print("IK path calculated.")
        print("Number of points in the path:", len(poses))
        if len(poses) > 0:
            print("IK path poses:", poses[current_centerline_index])
            print("IK path thetas:", thetas[current_centerline_index])
            print("IK path phis:", phis[current_centerline_index])
            print("IK path ds:", ds[current_centerline_index])

        # Instantiate your simulated robot logger
        initial_robot_position = [0, 0, 0] # Adjust as needed
        robot_logger = BroncoRobot2(initialPosition=initial_robot_position)
        logged_joint_values_forward = []

        last_valid_orientation = (0.0, 0.0, 0.0)

        euclidean_distances = []


        # Velocity and acceleration limits (translation in cm, angles in degrees)

        scaling_factor = 1.0  # Change to 0.5 or 2.0 as needed

        vmax_base = [1, 60, 7]
        vmax = [v * scaling_factor for v in vmax_base]  # [cm/s, deg/s, deg/s]

        scaling_factor_acc = 1.0  # Change to 0.5 for 50%, 2.0 for 200%, etc.

        amax_base = [100, 10, 10]
        amax = [a * scaling_factor_acc for a in amax_base]  # [cm/s², deg/s², deg/s²]

        dt = 1.0 / 120.0        # Simulation time step

        prev_joints = [ds[0], phis[0], thetas[0]]
        prev_vel = [0.0, 0.0, 0.0]

        while current_centerline_index < num_points - 1:
            tic = time.time()
            p.stepSimulation()

            if current_centerline_index < len(poses):
                target_pose = poses[current_centerline_index]
                t = target_pose[:3, 3] # Extract position

                # Log Cartesian position and orientation (Euler angles)
                position = target_pose[:3, 3]
                rotation_matrix = target_pose[:3, :3]
                euler_angles = RR.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)  # Requires scipy.spatial.transform.Rotation as RR

                cartesian_log_file.write(
                    f"Frame {frame_count} - Position: {position}, Euler Angles (deg): {euler_angles}\n"
                )

                closest_point_id = self.pointLocator.FindClosestPoint(t)
                closest_point = np.array(self.vtkdata.GetPoint(closest_point_id))
                distance_to_wall = np.linalg.norm(t - closest_point)
                collision_log_file.write(f"Frame {frame_count} - Closest point: {closest_point}, Distance to wall: {distance_to_wall}\n")
                # Check for collision with the airway wall
                collision_threshold = 0.005  # 5 mm, adjust as needed
                if distance_to_wall < collision_threshold:
                    print(f"Collision detected! Distance to wall: {distance_to_wall:.4f} m")
                    collision_log_file.write(f"Collision detected at index {current_centerline_index} with distance {distance_to_wall:.4f} m\n")

                # Get the desired joint values from IK using the correct index
                # Get target joints (angles in degrees)
                target_joints = [ds[current_centerline_index], phis[current_centerline_index], thetas[current_centerline_index]]
                joint_log_file_forward.write(f"Frame {frame_count} - Joint Data Forward: {target_joints}\n")

                # Compute desired velocity for each joint
                desired_vel = [(target_joints[i] - prev_joints[i]) / dt for i in range(3)]

                # Apply velocity and acceleration limits
                limited_vel = []
                for i in range(3):
                    v = np.clip(desired_vel[i], -vmax[i], vmax[i])
                    delta_v = v - prev_vel[i]
                    max_delta_v = amax[i] * dt
                    v = prev_vel[i] + np.clip(delta_v, -max_delta_v, max_delta_v)
                    limited_vel.append(v)

                # Compute new joint values
                #new_joints = [prev_joints[i] + limited_vel[i] * dt for i in range(3)]
                new_joints = target_joints
                #new_joints = prev_joints.copy()
                #joint_to_update = 2  # 0 for joint 1, 1 for joint 2, 2 for joint 3
                #new_joints[joint_to_update] = prev_joints[joint_to_update] + limited_vel[joint_to_update] * dt

                joint_log_file.write(f"Frame {frame_count} - Joint Data: {new_joints}\n")

                robot_logger.setJoints(new_joints)
                prev_joints = new_joints
                prev_vel = limited_vel

                # Only increment index if close enough to the target
                if np.allclose(new_joints, target_joints, atol=1e-2):  # Use a tolerance suitable for degrees
                    current_centerline_index += 1

                # Optionally, update current_robot_joints for logging
                current_robot_joints = robot_logger.getJoints()

                # Log the desired and actual joint values along with the corresponding centerline point
                #print(f"Step {i}:")
                #print(f"  Centerline Point: {self.centerlineArray[i]}")
                #print(f"  Desired Joints (d, phi, theta): {desired_robot_joints}")
                #print(f"  Actual Robot Joints: {current_robot_joints}") # added current robot joints
                #print(f"Forward - Step {current_centerline_index}: Desired Joints = {desired_robot_joints}")

                pitch_cam = 0.0
                roll_cam = 0.0
                yaw_cam = 0.0

                # Calculate desired camera orientation based on the current path segment
                if current_centerline_index < num_points - 1:
                    pos_vector_gt = (self.centerlineArray[current_centerline_index + 1] - self.centerlineArray[current_centerline_index])
                    pos_vector_norm = np.linalg.norm(pos_vector_gt)
                    if pos_vector_norm > 1e-6:
                        forward = pos_vector_gt / pos_vector_norm
                        ref_up = np.array([0, 0, 1])
                        right = np.cross(forward, ref_up)
                        right_norm = np.linalg.norm(right)
                        if right_norm > 1e-6:
                            right /= right_norm
                            up = np.cross(right, forward)
                            up_norm = np.linalg.norm(up)
                            if up_norm > 1e-6:
                                up /= up_norm
                                rotation_matrix = np.array([right, up, forward]).T
                                if np.all(np.isfinite(rotation_matrix)):
                                    try:
                                        camera_quat = dcm2quat(rotation_matrix)
                                        pitch_cam, roll_cam, yaw_cam = p.getEulerFromQuaternion(camera_quat)
                                        last_valid_orientation = (pitch_cam, roll_cam, yaw_cam)  # Update last valid
                                    except Exception as e:
                                        print(f"Invalid rotation matrix, using last valid. Error: {e}")
                                        pitch_cam, roll_cam, yaw_cam = last_valid_orientation
                                else:
                                    pitch_cam, roll_cam, yaw_cam = last_valid_orientation
                            else:
                                pitch_cam, roll_cam, yaw_cam = last_valid_orientation
                        else:
                            pitch_cam, roll_cam, yaw_cam = last_valid_orientation
                    else:
                        pitch_cam, roll_cam, yaw_cam = last_valid_orientation
                    forward_orientations.append((pitch_cam, roll_cam, yaw_cam))

                quatCam = p.getQuaternionFromEuler([pitch_cam + np.pi / 2, roll_cam, yaw_cam])
                R_currentCam = p.getMatrixFromQuaternion(quatCam)
                R_currentCam = np.reshape(R_currentCam, (3, 3))

                pose = np.identity(4)
                pose[:3, 3] = t
                pose[:3, :3] = R_currentCam

                # Get Images from the current pose
                if current_centerline_index < num_points - 1:
                    pos_vector = self.centerlineArray[current_centerline_index + 1] - self.centerlineArray[current_centerline_index]
                    pos_vector /= np.linalg.norm(pos_vector) if np.linalg.norm(pos_vector) > 0 else np.array([0, 0, 1])
                else:
                    pos_vector = np.array([0, 0, 1])

                pose_img, rgb_img, depth_img, rgb_img_ori, depth_img2, depth_img3 = self.get_imagesPRY(
                    yaw_cam / np.pi * 180, pitch_cam / np.pi * 180, roll_cam / np.pi * 180, t, pos_vector
                )

                self.update_count += 1

                 # Log how far the robot has traveled
                start_index = 0  # if you start at the beginning
                current_index = current_centerline_index

                if current_index > start_index:
                    traversed_length = np.sum(
                        np.linalg.norm(
                            np.diff(self.originalCenterlineArray[start_index:current_index + 1], axis=0),
                            axis=1
                        )
                    )
                else:
                    traversed_length = 0  # not moved yet

                percent_traversed = traversed_length / self.centerline_length * 100
                current_centerline_length = np.sum(
                    np.linalg.norm(
                        np.diff(self.originalCenterlineArray[start_index:current_index + 1], axis=0),
                        axis=1
                    )
                )
                centerline_traversed_log_file.write(
                    f"Frame {frame_count} - Traversed centerline length: {traversed_length:.4f} ({percent_traversed:.2f}%)\n"
                )

                ## Construction of point cloud

                #construct the translation matrix
                translation_difference = t-T_current #Calculate translation difference

                # Construct the transformation matrix
                transformation_matrix = self.get_transformation_matrix(R_currentCam, T_current)  # Use translation difference

                #log joint data
                #joint_data = path_joint, path_jointvel
                # Save joint data to the log file
                #joint_log_file.write(f"Frame {frame_count} - Joint Data: {joint_data}\n")

                # Update point cloud every x frames


                frame_count += 1
                if frame_count % 5 == 0:
                    if depth_img2 is not None and np.any(depth_img2 > 0):
                        # Step 1: Process depth and pose
                        point_cloud_generator.process_depth_and_transform(depth_img2, pose_img)

                        # Step 2: Log transformation and pose
                        transformation_log_file.write(np.array_str(transformation_matrix) + "\n")
                        pose_log_file.write(np.array_str(pose_img) + "\n")

                        # Step 3: Save intermediate and aligned point cloud
                        temp_pcd = point_cloud_generator.save_intermediate_pointcloud(
                            depth_img2, transformation_matrix, f"intermediate_point_cloud_{frame_count}"
                        )
                        #if temp_pcd is not None:
                        #    print(f"Frame {frame_count}: temp_pcd has {len(temp_pcd.points)} points.")

                        #    aligned_filename = f"aligned_point_cloud_{frame_count}.ply"
                        #    output_folder = os.path.join("pointclouds", "aligned_pointclouds")
                        #    os.makedirs(output_folder, exist_ok=True)
                        #    aligned_file_path = os.path.join(output_folder, aligned_filename)
                        #    o3d.io.write_point_cloud(aligned_file_path, temp_pcd)

                        # Step 4: Compute centerline using CenterlineGenerator
                        #print(f"Computing and saving centerline for frame {frame_count}...")
                        #points, lines = centerline_generator.compute_and_save_centerline(
                        #    temp_pcd, output_filename=f"centerline_{frame_count}.ply"
                        #)

                        #if points is not None and lines is not None:
                            # Step 5: Accumulate centerline points and lines
                            #if accumulated_points is None:
                                # accumulated_points = points
                                # accumulated_lines = lines
                            #else:
                                # offset = len(accumulated_points)
                                # adjusted_lines = [[i + offset, j + offset] for i, j in lines]
                                # accumulated_points = np.vstack([accumulated_points, points])
                                # accumulated_lines.extend(adjusted_lines)

                        if self.originalCenterlineArray  is not None:

                            # Step 6: Save the accumulated centerline after processing every 10 frames
                            if frame_count % 20 == 0:
                                #print(f"Saving accumulated centerline at frame {frame_count}...")
                                #accumulated_filename = f"accumulated_centerline_{frame_count}.ply"
                                #centerline_generator.save_centerline_with_lines(accumulated_points, accumulated_lines, accumulated_filename)

                                # Step 7: Log the kinematics results with clear naming
                                kinematic_log_file.write(f"Frame {frame_count} Kinematics Results (Target Pose):\n")
                                kinematic_log_file.write(f" Position: {t}\n")
                                kinematic_log_file.write(f" Euler Angles (XYZ - pitch, roll, yaw): {pitch_cam:.6f}, {roll_cam:.6f}, {yaw_cam:.6f}\n")
                                kinematic_log_file.write(f" Transformation Matrix:\n{target_pose}\n")

                                print(f"Frame {frame_count} - Target Position: {t}")
                                print(f"Frame {frame_count} - Target Euler (PRY): {pitch_cam:.6f}, {roll_cam:.6f}, {yaw_cam:.6f}")

                                # Calculate and print pose difference
                                translation_diff = target_pose[:3, 3] - pose_img[:3, 3]
                                translation_diff_norm = np.linalg.norm(translation_diff) # Magnitude of the error vector
                                target_pose_norm = np.linalg.norm(target_pose[:3, 3])   # Magnitude of the target position vector

                                # Log deviation on each axis
                                print(f"Translation deviation (x, y, z): {translation_diff}")
                                kinematic_deviation_log_file.write(
                                    f"Frame {frame_count} - Translation deviation (x, y, z): {translation_diff}\n")

                                abs_translation_diff = np.abs(translation_diff)
                                print(f"Absolute translation deviation (x, y, z): {abs_translation_diff}")
                                kinematic_deviation_log_file.write(
                                    f"Frame {frame_count} - Absolute translation deviation (x, y, z): {abs_translation_diff}\n"
                                )

                                # Extract and correct actual pose from pose_img
                                R_target = target_pose[:3, :3]

                                # --- Apply pitch correction ---
                                R_current = pose_img[:3, :3]
                                R_current_euler = RR.from_matrix(R_current).as_euler('xyz')
                                R_current_euler[0] -= np.pi / 2  # Undo +π/2 pitch compensation
                                R_corrected = RR.from_euler('xyz', R_current_euler).as_matrix()

                                # Recompute pose_img with corrected orientation
                                corrected_pose_img = np.copy(pose_img)
                                corrected_pose_img[:3, :3] = R_corrected

                                # Diagnostic and logging using corrected pose
                                R_diff = R_target @ R_corrected.T
                                rot_diff = RR.from_matrix(R_diff)
                                euler_diff = rot_diff.as_euler('xyz', degrees=True)
                                euler_diff_norm = np.linalg.norm(euler_diff)

                                print_diagnostic(target_pose, corrected_pose_img, label=f"Frame {frame_count}")
                                kinematic_deviation_log_file.write(
                                    f"Frame {frame_count} - Rotation Difference (Euler angles): {euler_diff} (magnitude: {euler_diff_norm:.2f}°)\n"
                                )

                    else:
                        print(f"Depth image is invalid at frame {frame_count}. Skipping point cloud update and save.")
                # Inside your main simulation loop, after you get t (the robot's position):
                if self.originalCenterlineArray is not None:
                    dists = np.linalg.norm(self.originalCenterlineArray - t, axis=1)
                    mean_dist = np.mean(dists)
                    max_dist = np.max(dists)
                    euclidean_distances.append(mean_dist)
                    centerline_euclidean_distance_log_file.write(f"Frame {frame_count} - Euclidean mean distance: {mean_dist:.6f} m\n")
                    centerline_euclidean_distance_log_file.write(f"Frame {frame_count} - Euclidean max distance: {max_dist:.6f} m\n")

                # Only save at certain intervals
                if frame_count % 5 == 0:  # Adjust the interval as needed
                    #print(f"Saving intermediate point cloud at step {frame_count}")
                    point_cloud_generator.save_intermediate_pointcloud(depth_img2, transformation_matrix, f"intermediate_point_cloud_{frame_count}")
                    #print(f"Intermediate point cloud saved at step {frame_count}")

                    #print(f"Saving accumulated point cloud at frame {frame_count}")
                    point_cloud_generator.save_accumulated_point_cloud("accumulated_point_cloud")
                    #print(f"Accumulated point cloud saved at frame {frame_count}")

                T_current = t.copy()
                R_current = R_currentCam.copy()

                cv2.imshow("rgb", np.transpose(rgb_img, axes=(1, 2, 0))[:, :, ::-1])
                cv2.imshow("depth", depth_img)

                current_centerline_index += 1
                toc = time.time()
                print("Step frequency:", 1 / (toc - tic))

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                print("Reached the end of the centerline (target).")

            print("Forward path complete. Logging reversed motion.")
    

        # Check if we reached a point close to the end of the forward path
        if current_centerline_index >= num_points - 5:  # Adjust the threshold as needed
            print("Initiating reversal.")
            reversed_centerline = np.flip(self.centerlineArray, axis=0)
            num_points_reversed = len(reversed_centerline)
            current_centerline_index_reversed = 0
            poses_reversed, thetas_reversed, phis_reversed, ds_reversed = kinem.inverse_kinematics_path(reversed_centerline)
            reversed_orientations = list(reversed(forward_orientations))
            logged_joint_values_backward = []

            while current_centerline_index_reversed < num_points_reversed - 1:
                tic = time.time()
                p.stepSimulation() # Keep the simulation stepping

                if current_centerline_index_reversed < len(poses_reversed):
                    target_pose_reversed = poses_reversed[current_centerline_index_reversed]
                    t_reversed = target_pose_reversed[:3, 3] # Extract position

                    # Calculate desired "joint" values for the reversed path
                    # Calculate desired "joint" values for the reversed path using the correct index
                    # Get target joints (angles in degrees)
                    target_joints = [ds[current_centerline_index], phis[current_centerline_index], thetas[current_centerline_index]]
                    joint_log_file_forward.write(f"Frame {frame_count} - Joint Data Forward: {target_joints}\n")

                    # Compute desired velocity for each joint
                    desired_vel = [(target_joints[i] - prev_joints[i]) / dt for i in range(3)]

                    # Apply velocity and acceleration limits
                    limited_vel = []
                    for i in range(3):
                        v = np.clip(desired_vel[i], -vmax[i], vmax[i])
                        delta_v = v - prev_vel[i]
                        max_delta_v = amax[i] * dt
                        v = prev_vel[i] + np.clip(delta_v, -max_delta_v, max_delta_v)
                        limited_vel.append(v)

                    # Compute new joint values
                    new_joints = [prev_joints[i] + limited_vel[i] * dt for i in range(3)]

                    robot_logger.setJoints(new_joints)
                    prev_joints = new_joints
                    prev_vel = limited_vel

                    # Only increment index if close enough to the target
                    if np.allclose(new_joints, target_joints, atol=1e-2):  # Use a tolerance suitable for degrees
                        current_centerline_index += 1

                    # Optionally, update current_robot_joints for logging
                    current_robot_joints = robot_logger.getJoints()

                    # Get the current robot state (joint values)
                    current_robot_joints = robot_logger.getJoints()
                    #print(f"Backward - Step {current_centerline_index_reversed}: Desired Joints = {desired_robot_joints_reversed}")

                    # Use the reversed orientation from the forward pass
                    if current_centerline_index_reversed < len(reversed_orientations):
                        pitch_cam_reversed, roll_cam_reversed, yaw_cam_reversed = reversed_orientations[current_centerline_index_reversed]
                    else:
                        pitch_cam_reversed, roll_cam_reversed, yaw_cam_reversed = 0.0, 0.0, 0.0 # Default if no orientation

                    quatCam_reversed = p.getQuaternionFromEuler([pitch_cam_reversed + np.pi / 2, roll_cam_reversed, yaw_cam_reversed])
                    R_currentCam_reversed = p.getMatrixFromQuaternion(quatCam_reversed)
                    R_currentCam_reversed = np.reshape(R_currentCam_reversed, (3, 3))

                    pose_img_reversed = np.identity(4)
                    pose_img_reversed[:3, 3] = t_reversed
                    pose_img_reversed[:3, :3] = R_currentCam_reversed

                    # Get Images for the reversed motion - the direction vector will be backward
                    if current_centerline_index_reversed < num_points_reversed - 1:
                        pos_vector_reversed = reversed_centerline[current_centerline_index_reversed + 1] - reversed_centerline[current_centerline_index_reversed]
                        pos_vector_reversed /= np.linalg.norm(pos_vector_reversed) if np.linalg.norm(pos_vector_reversed) > 0 else np.array([0, 0, -1])
                    else:
                        pos_vector_reversed = np.array([0, 0, -1])

                    pose_img, rgb_img, depth_img, rgb_img_ori, depth_img2, depth_img3 = self.get_imagesPRY(
                        yaw_cam_reversed / np.pi * 180, pitch_cam_reversed / np.pi * 180, roll_cam_reversed / np.pi * 180, t_reversed, -pos_vector_reversed
                    )

                    cv2.imshow("rgb", np.transpose(rgb_img, axes=(1, 2, 0))[:, :, ::-1])
                    cv2.imshow("depth", depth_img)

                    current_centerline_index_reversed += 1
                    toc = time.time()
                    print("Backward Step frequency:", 1 / (toc - tic))

                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break

                else:
                    print("Reached the start of the centerline.")
                    current_centerline_index = num_points - 1
                    print("\nForward Joint Values logged")
                    for values in logged_joint_values_forward:
                        joint_log_file.write(f"Forward Joint Values: {values}\n")
                    print("\nBackward Joint Values logged:")
                    for values in logged_joint_values_backward:
                        joint_log_file.write(f"Backward Joint Values: {values}\n")
                    break

            else:

                print("Forward loop finished prematurely. Reversal not initiated.")
                print("\nForward Joint Values logged")
                for values in logged_joint_values_forward:
                    joint_log_file.write(f"Forward Joint Values: {values}\n")
                print("\nBackward Joint Values logged:")
                for values in logged_joint_values_backward:
                    joint_log_file.write(f"Backward Joint Values: {values}\n")
                
                
        
        transformation_log_file.close()
        pose_log_file.close()
        translation_log_file.close()
        translation_difference_log_file.close()
        rotation_log_file.close()
        rotation_log_file_2.close()
        kinematic_log_file.close()
        kinematic_deviation_log_file.close()
        centerline_traversed_log_file.close()
        
        

        #cv2.destroyAllWindows()
        #p.disconnect()


        return path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, self.originalCenterlineArray, safe_distance_list, path_jointvel, path_joint
    
    

        # --- Diagnostic: Compare Rotation Matrices, Quaternions, Euler Angles ---

