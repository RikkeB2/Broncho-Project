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

from .camera import fixedCamera
from .keyBoardEvents import getDirectionBO, getAdditionBO

from scipy.io import savemat

from .simRobot import BroncoRobot1 

from .pointCloudGenerator import PointCloudGenerator

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

        mesh = reader.GetOutput()
        points = mesh.GetPoints()
        data = points.GetData()
        centerlineArray = vtk_to_numpy(data)
        centerlineArray = np.dot(self.R_model, centerlineArray.T).T * 0.01 + self.t_model

        # Downsample or upsample the centerline to the same length/size rate
        centerline_length = 0
        for i in range(len(centerlineArray) - 1):
            length_diff = np.linalg.norm(centerlineArray[i] - centerlineArray[i + 1])
            centerline_length += length_diff
        centerline_size = len(centerlineArray)
        lenth_size_rate = 0.007  # refer to Siliconmodel1
        centerline_size_exp = int(centerline_length / lenth_size_rate)
        centerlineArray_exp = np.zeros((centerline_size_exp, 3))
        for index_exp in range(centerline_size_exp):
            index = index_exp / (centerline_size_exp - 1) * (centerline_size - 1) # calcula a percentagem de caminho em que esta e vai procurar qual o index correspondente no path inicial
            index_left_bound = int(index) # vai buscar o valor a esquerda e a direita
            index_right_bound = int(index) + 1
            if index_left_bound == centerline_size - 1:
                centerlineArray_exp[index_exp] = centerlineArray[index_left_bound]
            else:
                centerlineArray_exp[index_exp] = (index_right_bound - index) * centerlineArray[index_left_bound] + (index - index_left_bound) * centerlineArray[index_right_bound] # calculate the medium value between the boundaries of them
        centerlineArray = centerlineArray_exp

        # Smoothing trajectory
        self.originalCenterlineArray = centerlineArray
        centerlineArray_smoothed = np.zeros_like(centerlineArray)
        for i in range(len(centerlineArray)):
            left_bound = i - 10
            right_bound = i + 10
            if left_bound < 0: left_bound = 0
            if right_bound > len(centerlineArray): right_bound = len(centerlineArray)
            centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound : right_bound], axis=0)
        self.centerlineArray = centerlineArray_smoothed

        # Calculate trajectory length
        centerline_length = 0
        for i in range(len(self.centerlineArray) - 1):
            length_diff = np.linalg.norm(self.centerlineArray[i] - self.centerlineArray[i + 1])
            centerline_length += length_diff
        self.centerline_length = centerline_length

        # Generate new path in each step
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.airway_model_dir) #modelo do pulmao oco (Hollow lung model)
        reader.Update()
        self.vtkdata = reader.GetOutput()
        self.targetPoint = centerlineArray[0]
        self.transformed_target = np.dot(np.linalg.inv(self.R_model), self.targetPoint - self.t_model) * 100 # recalcula para o ponto inicial 
        self.transformed_target_vtk_cor = np.array([-self.transformed_target[0], -self.transformed_target[1], self.transformed_target[2]])  #put in the same coordinate system # x and y here is opposite to those in the world coordinate system
        
        # Collision detection
        self.pointLocator = vtk.vtkPointLocator()
        self.pointLocator.SetDataSet(self.vtkdata)
        self.pointLocator.BuildLocator()

        self.camera = fixedCamera(0.01, p)
        #self.camera.width = 800
        #self.camera.height = 800

        boundingbox = p.getAABB(airwayBodyId)
        print(boundingbox)
        print(np.max(centerlineArray, axis=0))
        print(np.min(centerlineArray, axis=0))
        print(np.argmax(centerlineArray, axis=0))
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

        # Debugging: Print depth image stats before normalization
        print(f"Depth Image Stats before normalization: Min={np.min(depth_img)}, Max={np.max(depth_img)}, Mean={np.mean(depth_img)}")

        depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img)) 
        depth_img = depth_img * 255
        depth_img = depth_img.astype(np.uint8)
        depth_img = cv2.resize(depth_img, (200, 200))

        # Debugging: Print depth image stats after normalization
        print(f"Depth Image Stats after normalization: Min={np.min(depth_img)}, Max={np.max(depth_img)}, Mean={np.mean(depth_img)}")

        return rgb_img, depth_img, rgb_img_ori, depth_img2

    def runManual(self, args, point_cloud_generator):

        count = len(self.centerlineArray) - 1

        start_index = len(self.centerlineArray) - 3
        start_index = len(self.centerlineArray) - 1

        x, y, z = self.centerlineArray[start_index]
        yaw = 0
        pitch = 0
        roll = 0

        ######################
        # Defining initial camera orientation to be always aligned with the path
        pos_vector_gt = (self.centerlineArray[count - 1] - self.centerlineArray[count]) / np.linalg.norm(self.centerlineArray[count - 1] - self.centerlineArray[count]) # vetor da direcao ideal do movimento
        pos_vector_norm = np.linalg.norm(pos_vector_gt)

        pitch = np.arcsin(pos_vector_gt[2] / pos_vector_norm)
        if pos_vector_gt[0] > 0:
            yaw = -np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))  
        else:
            yaw = np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))
        ##############################

        #pitch, yaw, x, y, z = self.random_start_point(rand_index=start_index) # em teste nao ha random point, comeca no inicio

        quat_init = p.getQuaternionFromEuler([pitch, roll, yaw])
        R = p.getMatrixFromQuaternion(quat_init)
        R = np.reshape(R, (3, 3))
        quat = dcm2quat(R)
        t = np.array([x, y, z])  # pode nao estar em cima do path porque foi adicionado aqui um random em torno do mesmo
        pos_vector = self.centerlineArray[count - 1] - self.centerlineArray[count]
        pos_vector_last = pos_vector

        for i in range(len(self.centerlineArray) - 1):
            p.addUserDebugLine(self.centerlineArray[i], self.centerlineArray[i + 1], lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3)
        
        path_length = 0
        path_centerline_error_list = []
        path_centerline_length_list = []
        path_centerline_ratio_list = []
        safe_distance_list = []
        path_centerline_pred_position_list = []

        path_trajectoryT = []
        path_trajectoryR = []

        ###
        m_distBO = 0.003 # distancia a percorrer em cada iteracao mm
        m_initalIdx = np.linalg.norm(self.originalCenterlineArray - t, axis=1).argmin()

        args.human = True
        direction = np.array([0, 0, 0])  

        frame_count = 0

        while 1:
            tic = time.time()
            p.stepSimulation()

            #0 - Pose atual da camara no mundo
            pitch_current = pitch
            yaw_current = yaw
            roll_current = roll
            quat_current = p.getQuaternionFromEuler([pitch_current, roll_current, yaw_current])
            R_current = p.getMatrixFromQuaternion(quat_current)
            R_current = np.reshape(R_current, (3, 3))
            T_current = t

            path_trajectoryT.append(t)
            path_trajectoryR.append(p.getMatrixFromQuaternion(quat_current))

            # Get Images from current pose
            rgb_img, depth_img, rgb_img_ori, depth_img2 =  self.get_imagesPRY(yaw / np.pi * 180, pitch / np.pi * 180, roll / np.pi * 180, t, pos_vector)
            print(f"Depth Image Shape: {depth_img2.shape}")
            print(f"Depth Image Sample: {depth_img2[100, 100]}")  # Check the value at a specific pixel
            print(f"Depth Image Stats: Min={np.min(depth_img2)}, Max={np.max(depth_img2)}, Mean={np.mean(depth_img2)}")

            #  Update point cloud every 10 frames
            frame_count += 1
            if frame_count % 50 == 0:
                if depth_img2 is not None and np.any(depth_img2 > 0):
                    point_cloud_generator.update_point_cloud(depth_img2)

                    # Only visualize & save at **certain intervals**
                    if frame_count % 100 == 0:  # Adjust as needed
                        print(f"Saving intermediate point cloud at step {frame_count}")
                        point_cloud_generator.save_pc(os.path.join("pointclouds", f"intermediate_point_cloud_{frame_count}.pcd"))
                        print(f"Intermediate point cloud saved at step {frame_count}")

                else:
                    print("Depth image is invalid. Skipping point cloud update and save.")

            # Visualization step (outside the loop)
            #if args.human and frame_count % 400 == 0:  # Show only every 100 frames
               # print("Showing point cloud... Press ESC or C to close.")
                #point_cloud_generator.show()  # Blocks execution until user closes the window

            # Get the nearest point of the center line to the current one
            nearest_original_centerline_point_sim_cor_index = np.linalg.norm(self.originalCenterlineArray - t, axis=1).argmin()

            # Stops the simulation if the index of the nearest point is closer to the end of the trajectory
            if nearest_original_centerline_point_sim_cor_index <= 10:  # reach the target point
                path_centerline_ratio_list.append(1.0)  # complete the path
                break

            # Manual input to the system 
            if args.human:
                direction = np.array([0, 0, 0])
                while(np.abs(direction).sum() == 0):
                    keys = p.getKeyboardEvents()
                    direction = getAdditionBO(keys, 1)
                    direction = np.array(direction)
                print("Direction:", direction)

            quatCam = p.getQuaternionFromEuler([pitch + np.pi / 2, roll, yaw])
            R_currentCam = p.getMatrixFromQuaternion(quatCam)
            R_currentCam = np.reshape(R_currentCam, (3, 3))
            
            # Implementar IK de acordo com os inputs dados        
            t =  t + np.dot(R_currentCam, [0, 0,  -direction[2] * 0.005])  # andar em Z na direcao da camara; OBS: o eixo da camara esta rodado em relacao ao mundo

            #quat_step = p.getQuaternionFromEuler([np.radians(direction[1]), 0, np.radians(direction[0])]) # pitch 0 yaw
            quat_step = p.getQuaternionFromEuler([np.radians(direction[1]), np.radians(direction[0]), 0]) # pitch roll 0

            
            R_step = p.getMatrixFromQuaternion(quat_step)
            R_step = np.reshape(R_step, (3, 3))

            quat_current = dcm2quat(np.dot(R_current,R_step))
            [pitch, roll, yaw] = p.getEulerFromQuaternion(quat_current) # Euler angles given as XYZ (pitch;roll;yaw)

            ########
            #pitch = pitch  + np.radians(direction[1])
            #yaw = yaw + np.radians(direction[0])
            #roll = roll + np.radians(direction[0])

            # Caluculate path length
            path_length_diff = np.linalg.norm(t-T_current)
            path_length += path_length_diff
            
            count -= 1
            toc = time.time()
            print("Step frequency:", 1 / (toc - tic))

            rgb_img = np.transpose(rgb_img, axes=(1, 2, 0))
            rgb_img = rgb_img[:, :, ::-1]  # RGB to BGR for showing
            cv2.imshow("rgb", rgb_img)
            cv2.imshow("depth", depth_img)
            cv2.imwrite("depth.png", depth_img)
            
            # 3 - Add info to guide
            # Get the nearest point
            if nearest_original_centerline_point_sim_cor_index <= 10:  # reach the target point
                path_centerline_ratio_list.append(1.0)  # complete the path
                break
            else:
                restSmoothedCenterlineArray = self.originalCenterlineArray[:nearest_original_centerline_point_sim_cor_index] 
                #restSmoothedCenterlineArray = self.originalCenterlineArray
                #index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, m_distBO) # vai procurar o ponto do path a seguir, a distancia de m_vel
                #index_form_dis2 = self.indexFromDistance(restSmoothedCenterlineArray[:index_form_dis], len(restSmoothedCenterlineArray[:index_form_dis]) - 1, m_distBO) # vai procurar o ponto do path a seguir, a distancia de m_vel
                index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, 0.01) # vai procurar o ponto do path a seguir, a distancia de m_vel
                index_form_dis2 = self.indexFromDistance(restSmoothedCenterlineArray[:index_form_dis], len(restSmoothedCenterlineArray[:index_form_dis]) - 1, 0.1) # vai procurar o ponto do path a seguir, a distancia de m_vel
 
                if not index_form_dis or not index_form_dis2:
                    index_form_dis = len(restSmoothedCenterlineArray) - 1
                    index_form_dis2 = len(restSmoothedCenterlineArray) - 2
                pos_vector_gt = (restSmoothedCenterlineArray[index_form_dis2] - restSmoothedCenterlineArray[index_form_dis]) / np.linalg.norm(restSmoothedCenterlineArray[index_form_dis2] - restSmoothedCenterlineArray[index_form_dis]) # vetor da direcao ideal do movimento


            pos_vector_norm = np.linalg.norm(pos_vector_gt)
            if pos_vector_norm < 1e-5:
                count -= 1
                continue
        
            # A - Proxima posicao camara
            m_nextgtpos = restSmoothedCenterlineArray[index_form_dis]
            #OU smooth da atual ate a proxima x 2
            m_nextgtposSmooth = np.mean(restSmoothedCenterlineArray[index_form_dis2-1:index_form_dis], axis=0) 

            # Get the direction to follow, getting the point in front in the trajectory path
            intrinsic_matrix = np.array([[175 / 1.008, 0, 100],
                                    [0, 175 / 1.008, 100],
                                    [0, 0, 1]])

            m_image = rgb_img.copy()

            pose = np.identity(4)
            pose[:3, 3] = T_current
            pose[:3, :3] = R_currentCam

            predicted_action_in_camera_cor = np.dot(np.linalg.inv(pose), [m_nextgtpos[0],m_nextgtpos[1],m_nextgtpos[2],1])[:-1]
            predicted_action_in_image_cor = np.dot(intrinsic_matrix, predicted_action_in_camera_cor) / predicted_action_in_camera_cor[2]

            predicted_action_in_camera_cor1 = np.dot(np.linalg.inv(pose), [m_nextgtposSmooth[0],m_nextgtposSmooth[1],m_nextgtposSmooth[2],1])[:-1]
            predicted_action_in_image_cor1 = np.dot(intrinsic_matrix, predicted_action_in_camera_cor1) / predicted_action_in_camera_cor1[2]

        
            cv2.circle(m_image, (100, 100), 3, (0, 0, 255), -1)
            #cv2.circle(m_image, (int(200-predicted_action_in_image_cor[0]), int(predicted_action_in_image_cor[1])), 3, (0, 255, 0), -1) # X direction is switched in image view
            cv2.circle(m_image, (int(200-predicted_action_in_image_cor1[0]), int(predicted_action_in_image_cor1[1])), 3, (255, 0, 0), -1) # X direction is switched in image view

            cv2.imshow("Image", m_image)

            cv2.waitKey(5)
        
        # Save the final point cloud after the loop
        print(f"Total points in point cloud before saving: {len(point_cloud_generator.pcd.points)}")
        if len(point_cloud_generator.pcd.points) > 0:
            print("Saving final point cloud...")
            point_cloud_generator.save_pc(os.path.join("pointclouds", "final_point_cloud.pcd"))
            print("Final point cloud saved.")
        else:
            print("Point cloud is empty. Nothing to save.")
        
        p.disconnect()
        self.r.delete()

        return path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, self.originalCenterlineArray, safe_distance_list


    def runVS2(self, args, point_cloud_generator):
        # Pitch and Roll    
        count = len(self.centerlineArray) - 1

        start_index = len(self.centerlineArray) - 3

        x, y, z = self.centerlineArray[start_index]
        yaw = 0
        pitch = 0
        roll = 0

        ######################
        # Defining initial camera orientation to be always aligned with the path
        pos_vector_gt = (self.centerlineArray[count - 1] - self.centerlineArray[count]) / np.linalg.norm(self.centerlineArray[count - 1] - self.centerlineArray[count]) # vetor da direcao ideal do movimento
        pos_vector_norm = np.linalg.norm(pos_vector_gt)

        pitch = np.arcsin(pos_vector_gt[2] / pos_vector_norm)
        if pos_vector_gt[0] > 0:
            yaw = -np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))  
        else:
            yaw = np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))
        ##############################

        quat_init = p.getQuaternionFromEuler([pitch, roll, yaw])
        R = p.getMatrixFromQuaternion(quat_init)
        R = np.reshape(R, (3, 3))
        t = np.array([x, y, z])  # pode nao estar em cima do path porque foi adicionado aqui um random em torno do mesmo
        pos_vector = self.centerlineArray[count - 1] - self.centerlineArray[count]

        for i in range(len(self.centerlineArray) - 1):
            p.addUserDebugLine(self.centerlineArray[i], self.centerlineArray[i + 1], lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3)
        
        path_length = 0
        path_centerline_ratio_list = []
        safe_distance_list = []

        path_trajectoryT = []
        path_trajectoryR = []

        path_joint = []
        path_jointvel = []

        ###
        m_distBO = 0.003 # distancia a percorrer em cada iteracao mm

        args.human = True
        direction = np.array([0, 0, 0]) 

        intrinsic_matrix = np.array([[175 / 1.008, 0, 100],
                        [0, 175 / 1.008, 100],
                        [0, 0, 1]]) 

        #Initialize robot
        m_robot = BroncoRobot1()

        frame_count = 0

        while 1:
            tic = time.time()
            p.stepSimulation()

            #0 - Pose atual da camara no mundo
            pitch_current = pitch
            yaw_current = yaw
            roll_current = roll
            quat_current = p.getQuaternionFromEuler([pitch_current, roll_current, yaw_current])
            R_current = p.getMatrixFromQuaternion(quat_current)
            R_current = np.reshape(R_current, (3, 3))
            T_current = t

            quatCam = p.getQuaternionFromEuler([pitch + np.pi / 2, roll, yaw])
            R_currentCam = p.getMatrixFromQuaternion(quatCam)
            R_currentCam = np.reshape(R_currentCam, (3, 3))

            pose = np.identity(4)
            pose[:3, 3] = T_current
            pose[:3, :3] = R_currentCam

            m_currentJoints = m_robot.getJoints()


            # Manual input to the system 
            keys = p.getKeyboardEvents()
            direction = getAdditionBO(keys, 1)
            direction = np.array(direction)       
            
            # Get Images from current pose
            rgb_img, depth_img, rgb_img_ori, depth_img2 =  self.get_imagesPRY(yaw / np.pi * 180, pitch / np.pi * 180, roll / np.pi * 180, t, pos_vector)

            print(f"Depth Image Shape: {depth_img2.shape}")
            print(f"Depth Image Sample: {depth_img2[100, 100]}")  # Check the value at a specific pixel
            print(f"Depth Image Stats: Min={np.min(depth_img2)}, Max={np.max(depth_img2)}, Mean={np.mean(depth_img2)}")

            #  Update point cloud every 10 frames
            frame_count += 1
            if frame_count % 50 == 0:
                if depth_img2 is not None and np.any(depth_img2 > 0):
                    point_cloud_generator.update_point_cloud(depth_img2)

                    # Only visualize & save at **certain intervals**
                    if frame_count % 100 == 0:  # Adjust as needed
                        print(f"Saving intermediate point cloud at step {frame_count}")
                        point_cloud_generator.save_pc(os.path.join("pointclouds", f"intermediate_point_cloud_{frame_count}.pcd"))
                        print(f"Intermediate point cloud saved at step {frame_count}")

                else:
                    print("Depth image is invalid. Skipping point cloud update and save.")

            # Visualization step (outside the loop)
            #if args.human and frame_count % 400 == 0:  # Show only every 100 frames
                #print("Showing point cloud... Press ESC or C to close.")
                #point_cloud_generator.show()  # Blocks execution until user closes the window


            # Get the nearest point of the center line to the current one
            nearest_original_centerline_point_sim_cor_index = np.linalg.norm(self.originalCenterlineArray - t, axis=1).argmin()

            # Stops the simulation if the index of the nearest point is closer to the end of the trajectory
            if nearest_original_centerline_point_sim_cor_index <= 10:  # reach the target point
                path_centerline_ratio_list.append(1.0)  # complete the path
                break
            else:
                restSmoothedCenterlineArray = self.originalCenterlineArray[:nearest_original_centerline_point_sim_cor_index] 
                index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, 0.01) # vai procurar o ponto do path a seguir, a distancia de m_vel
                index_form_dis2 = self.indexFromDistance(restSmoothedCenterlineArray[:index_form_dis], len(restSmoothedCenterlineArray[:index_form_dis]) - 1, 0.1) # vai procurar o ponto do path a seguir, a distancia de m_vel
                if not index_form_dis or not index_form_dis2:
                    index_form_dis = len(restSmoothedCenterlineArray) - 1
                    index_form_dis2 = len(restSmoothedCenterlineArray) - 2
                    
                pos_vector_gt = (restSmoothedCenterlineArray[index_form_dis2] - restSmoothedCenterlineArray[index_form_dis]) / np.linalg.norm(restSmoothedCenterlineArray[index_form_dis2] - restSmoothedCenterlineArray[index_form_dis]) # vetor da direcao ideal do movimento

            pos_vector_norm = np.linalg.norm(pos_vector_gt)
            if pos_vector_norm < 1e-5:
                count -= 1
                continue
        
            # A - Proxima posicao camara
            #OU smooth da atual ate a proxima x 2
            m_nextgtposSmooth = np.mean(restSmoothedCenterlineArray[index_form_dis2-1:index_form_dis], axis=0) 

            predicted_action_in_camera_cor1 = np.dot(np.linalg.inv(pose), [m_nextgtposSmooth[0],m_nextgtposSmooth[1],m_nextgtposSmooth[2],1])[:-1]
            predicted_action_in_image_cor1 = np.dot(intrinsic_matrix, predicted_action_in_camera_cor1) / predicted_action_in_camera_cor1[2]

            # VIsual servoing
            m_jointsVelRel, m_nextvalues = m_robot.visualservoingcontrol(predicted_action_in_image_cor1)
            m_nextvalues[0] = m_nextvalues[0] - direction[2] * 0.005
            
            quat_step = p.getQuaternionFromEuler([m_nextvalues[2] - m_currentJoints[2], m_nextvalues[1] - m_currentJoints[1], 0]) # pitch roll 0
            t =  t + np.dot(R_currentCam, [0, 0,  -direction[2] * 0.005])  # andar em Z na direcao da camara; OBS: o eixo da camara esta rodado em relacao ao mundo

            R_step = p.getMatrixFromQuaternion(quat_step)
            R_step = np.reshape(R_step, (3, 3))

            quat_current = dcm2quat(np.dot(R_current,R_step))
            [pitch, roll, yaw] = p.getEulerFromQuaternion(quat_current) # Euler angles given as XYZ (pitch;roll;yaw)

            ########
            # Caluculate path length
            path_length_diff = np.linalg.norm(t-T_current)
            path_length += path_length_diff 
            
            count -= 1
            toc = time.time()
            print("Step frequency:", 1 / (toc - tic))

            rgb_img = np.transpose(rgb_img, axes=(1, 2, 0))
            rgb_img = rgb_img[:, :, ::-1]  # RGB to BGR for showing
            cv2.imshow("rgb", rgb_img)
            cv2.imshow("depth", depth_img)
            
            # Get the direction to follow, getting the point in front in the trajectory path
            m_image = rgb_img.copy()

            cv2.circle(m_image, (100, 100), 3, (0, 0, 255), -1)
            cv2.circle(m_image, (int(200-predicted_action_in_image_cor1[0]), int(predicted_action_in_image_cor1[1])), 3, (255, 0, 0), -1) # X direction is switched in image view

            cv2.imshow("Image", m_image)

            #save information from the path
            path_trajectoryT.append(t)
            path_trajectoryR.append(p.getMatrixFromQuaternion(quat_current))
            path_jointvel.append(m_jointsVelRel)
            path_joint.append(m_nextvalues)

            cv2.waitKey(5)

        # Save the final point cloud after the loop
        print(f"Total points in point cloud before saving: {len(point_cloud_generator.pcd.points)}")
        if len(point_cloud_generator.pcd.points) > 0:
            print("Saving final point cloud...")
            point_cloud_generator.save_pc(os.path.join("pointclouds", "final_point_cloud.pcd"))
            print("Final point cloud saved.")
        else:
            print("Point cloud is empty. Nothing to save.")
        
        p.disconnect()
        self.r.delete()

        return path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, self.originalCenterlineArray, safe_distance_list, path_jointvel, path_joint

