import pybullet as p
import numpy as np
import os


class Camera(object):
    def __init__(self):
        pass
    
    def getViewMatrix(self):
        pass
    
    def getProjectionMatrix(self):
        pass

    def getImg(self):
        pass

    def getOrientation(self):
        return self.yaw, self.pitch, self.roll

    def getPosition(self):
        return self.targetPos


class fixedCamera(Camera):

    def __init__(self, dis, physics_server, targetPos = [0, 0, 0], physicsClientId = 0, 
                # RPY
                yaw=0, pitch=0, roll=0, upAxisIndex = 2,
                # Intrinsics
                fov=2 * np.arctan(100 / 181.9375) / np.pi * 180, aspect=None, nearVal=0.00001, farVal=100,
                # IMG
                width=200, height=200
                ):

        self.dis = dis
        self.targetPos = targetPos
        self.physicsClientId = physicsClientId
        self.p = physics_server

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.upAxisIndex = upAxisIndex

        self.fov = fov
        self.aspect = aspect
        if aspect is None:
            self.aspect = width / height
        self.far = farVal
        self.near = nearVal

        self.width = width
        self.height = height

        self.previous_position = self.getPosition()
        self.frame_count = 0

    def getViewMatrix(self):

        viewMatrix = self.p.computeViewMatrixFromYawPitchRoll(
            distance=self.dis, yaw=self.yaw, pitch=self.pitch, roll=self.roll, 
            upAxisIndex=self.upAxisIndex, cameraTargetPosition=self.targetPos,
            physicsClientId = self.physicsClientId
            )
        
        return viewMatrix

    def getProjectionMatrix(self):

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far,
            physicsClientId=self.physicsClientId
        )

        return projectionMatrix

    def getIntrinsic(self):

        projectionMatrix = self.getProjectionMatrix()
        projectionMatrix = np.reshape(projectionMatrix, [4, 4])
        fx = projectionMatrix[0, 0] * self.width / 2
        fy = projectionMatrix[1, 1] * self.height / 2
        intrinsic = np.zeros([3, 3])
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[2, 2] = 1

        return intrinsic

    def lookat(self, yaw, pitch, targetPos, lightDirection):
        
        self.yaw = yaw
        self.pitch = pitch
        self.targetPos = targetPos
        self.lightDirection = lightDirection

        return self.visualize()
    

    def lookatBO(self, yaw, pitch, roll, targetPos, lightDirection):
        
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.targetPos = targetPos
        self.lightDirection = lightDirection

        return self.visualize()

    def calculateTranslation(self):
        current_position = self.getPosition()
        translation = np.array(current_position) - np.array(self.previous_position)
        self.previous_position = current_position
        return translation

    def visualize(self):

        # self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1, lightPosition=[0, 0, 4])
        self.p.resetDebugVisualizerCamera(self.dis, self.yaw, self.pitch, self.targetPos)
        # print(self.targetPos)

        camera_info = self.p.getDebugVisualizerCamera(physicsClientId=self.physicsClientId)
        width = camera_info[0]
        height = camera_info[1]
        # viewMatrix = camera_info[2]
        # projectionMatrix = camera_info[3]
        viewMatrix = self.getViewMatrix()
        projectionMatrix = self.getProjectionMatrix()
        intrinsic = self.getIntrinsic()
        _, _, rgbImg, depthImg, segImg =\
             p.getCameraImage(self.width, self.height,
                            viewMatrix,
                            projectionMatrix,
                            lightDirection=self.lightDirection,
                            # lightDirection=[0, -1, 0],
                            # lightAmbientCoeff=0.5,
                            # lightDiffuseCoeff=0.5,
                            # lightDistance=0.1,
                            # shadow=1,
                            # lightSpecularCoeff=100,
                            # lightDirection=[0, 0, 10],
                            physicsClientId=self.physicsClientId)
        rgbImg = np.reshape(rgbImg, [200,200,4])
        depthImg = np.reshape(depthImg, [200,200])
        segImg = np.reshape(segImg, [200,200])

        # Print the current orientation
        yaw, pitch, roll = self.getOrientation()
        print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

        # Calculate and save translation every 100 timesteps
        if self.frame_count % 100 == 0:
            translation = self.calculateTranslation()
            with open("translation_log.txt", "a") as file:
                file.write(f"Translation: {translation}\n")

        self.frame_count += 1

        return rgbImg, depthImg, segImg


class movingCamera(Camera):

    def __init__(self, dis, physics_server, targetPos = [0, 0, 0], physicsClientId = 0, 
                # RPY
                yaw=0, pitch=-40, roll=0, upAxisIndex = 2,
                # Intrinsics
                fov=50, aspect=None, nearVal=0.01, farVal=10000,
                # IMG
                width=320, height=240
                ):

        self.dis = dis
        self.targetPos = targetPos
        self.physicsClientId = physicsClientId
        self.p = physics_server

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.upAxisIndex = upAxisIndex

        self.fov = fov
        self.aspect = aspect
        if aspect is None:
            self.aspect = width/height
        self.far = farVal
        self.near = nearVal

        self.width = width
        self.height = height

        self.previous_position = self.getPosition()
        self.frame_count = 0

    def getViewMatrix(self):

        viewMatrix = self.p.computeViewMatrixFromYawPitchRoll(
            distance=self.dis, yaw=self.yaw, pitch=self.pitch, roll=self.roll, 
            upAxisIndex=self.upAxisIndex, cameraTargetPosition=self.targetPos,
            physicsClientId = self.physicsClientId
            )
        
        return viewMatrix

    def getProjectionMatrix(self):

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=self.fov, 
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far, 
            physicsClientId=self.physicsClientId
        )

        return projectionMatrix

    def getImg(self):
        # get images
        viewMatrix = self.getViewMatrix()
        projectionMatrix = self.getProjectionMatrix()
        width, height, rgbImg, depthImg, segImg =\
             p.getCameraImage(self.width, self.height,
                            viewMatrix,
                            projectionMatrix,
                            # lightDirection=[-0.15, 0.05, 6],
                            # lightDirection=[0, 0, 10],
                            physicsClientId=self.physicsClientId)
        # postprocess
        depth = self.far * self.near / (self.far - (self.far - self.near) * depthImg)
        rgb = np.reshape(rgbImg, [height, width, 4])
        rgb = rgb[:, :, :3]
        depth = np.reshape(depth, [height, width])

        return rgb, depth

    def getIntrinsic(self):

        projectionMatrix = self.getProjectionMatrix()
        projectionMatrix = np.reshape(projectionMatrix, [4, 4])
        fx = projectionMatrix[0, 0] * self.width/2
        fy = projectionMatrix[1, 1] * self.height/2
        intrinsic = np.zeros([3, 3])
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[2, 2] = 1

        return intrinsic

    def lookat(self, yaw, pitch, targetPos):
        
        self.yaw = yaw
        self.pitch = pitch
        self.targetPos = targetPos

        self.visualize()

    def calculateTranslation(self):
        current_position = self.getPosition()
        translation = np.array(current_position) - np.array(self.previous_position)
        self.previous_position = current_position
        return translation

    def visualize(self):

        # self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1, lightPosition=[0, 0, 4])
        self.p.resetDebugVisualizerCamera(self.dis, self.yaw, self.pitch, self.targetPos)

        # Print the current orientation
        yaw, pitch, roll = self.getOrientation()
        print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

        # Calculate and save translation every 100 timesteps
        if self.frame_count % 100 == 0:
            translation = self.calculateTranslation()
            with open("translation_log.txt", "a") as file:
                file.write(f"Translation: {translation}\n")

        self.frame_count += 1