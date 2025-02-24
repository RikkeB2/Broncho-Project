import pybullet as p
import numpy as np

# ------------------------- Overall --------------------------------
# This code defines two classes for camera, fixedCamera and movingCamera.
# Both classes inherit from a base camera class, and provide methods for
# capturing images, getting intrinsics and adjusting the camera view
# ------------------------------------------------------------------


# Function is currently unused, is assumed to be a helper function for capturing images
# calculates view and projection matrix, and then ues p.getCameraImage to get the image data.
def setCameraPicAndGetPic(RPY = True,
                        cameraEyePosition=[0, 1, 1], cameraUpVector=[0,-1,0],
                        distance=0.5, yaw=0, pitch=-30, roll = 0, upAxisIndex = 2, 
                        cameraTargetPosition=[0, 0, 0], 
                        width : int = 320, height : int = 240, physicsClientId : int = 0):
    """

    """
    # basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)

    # basePos = np.array(basePos)
    # cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
    targetPos = cameraPos + 1 * tx_vec

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=50.0,               
        aspect=1.0,
        nearVal=0.01,            
        farVal=20,               
        physicsClientId=physicsClientId
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )
    
    return width, height, rgbImg, depthImg, segImg

# Base class for camera, serves as parent for fixedCamera and movingCamera
class Camera(object):
    def __init__(self):
        pass
    
    def getViewMatrix(self):
        pass
    
    def getProjectionMatrix(self):
        pass

    def getImg(self):
        pass


# Class representing camera with fixed position and orientation
class fixedCamera(Camera):

    def __init__(self, dis, physics_server, targetPos = [0, 0, 0], physicsClientId = 0, 
                # RPY
                yaw=0, pitch=0, roll=0, upAxisIndex = 2,
                # Intrinsics, euler angles defining orientation
                fov=2 * np.arctan(100 / 181.9375) / np.pi * 180, aspect=None, nearVal=0.00001, farVal=100,
                # Image specifications 200x200 pixels
                width=200, height=200
                ):

        self.dis = dis                          # Distance from target
        self.targetPos = targetPos              # Target position
        self.physicsClientId = physicsClientId  # ID of physics client
        self.p = physics_server                 # Physics server

        self.yaw = yaw                          # Yaw angle
        self.pitch = pitch                      # Pitch angle
        self.roll = roll                        # Roll angle
        self.upAxisIndex = upAxisIndex          # Up axis index (usually 2 for z)

        self.fov = fov                          # Field of view
        self.aspect = aspect                    # Aspect ratio of image
        if aspect is None:                      # If aspect ratio is not provided, calculate it
            self.aspect = width / height        # Aspect ratio is width divided by height
        self.far = farVal                       # Far clipping plane
        self.near = nearVal                     # Near clipping plane

        self.width = width
        self.height = height

    # Computes matrix, and transforms world coordinates into camera coordinates
    def getViewMatrix(self):

        viewMatrix = self.p.computeViewMatrixFromYawPitchRoll(
            distance=self.dis, yaw=self.yaw, pitch=self.pitch, roll=self.roll, 
            upAxisIndex=self.upAxisIndex, cameraTargetPosition=self.targetPos,
            physicsClientId = self.physicsClientId
            )
        
        return viewMatrix

    # Computes projection matrix, which projects 3D points onto the 2D image plane
    def getProjectionMatrix(self):

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far,
            physicsClientId=self.physicsClientId
        )

        return projectionMatrix

    # Captures image from camera
    def getImg(self):
        # Get the necessary matrices, extract the image data
        viewMatrix = self.getViewMatrix()
        projectionMatrix = self.getProjectionMatrix()
        width, height, rgbImg, depthImg, segImg =\
             p.getCameraImage(self.width, self.height,
                            viewMatrix,
                            projectionMatrix,
                            # lightDirection=[-0.15, 0.05, 6],
                            # lightDirection=[0, 0, 10],
                            physicsClientId=self.physicsClientId)
        # Postprocess to convert to actual distance
        depth = self.far * self.near / (self.far - (self.far - self.near) * depthImg)
        rgb = np.reshape(rgbImg, [height, width, 4])
        rgb = rgb[:, :, :3]
        depth = np.reshape(depth, [height, width])

        return rgb, depth

    # Calculate intrinsic matrix (focal length) from projection matrix
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

    # Adjusts camera view to given yaw, pitch and target position
    def lookat(self, yaw, pitch, targetPos, lightDirection):
        
        self.yaw = yaw
        self.pitch = pitch
        self.targetPos = targetPos
        self.lightDirection = lightDirection

        return self.visualize()
    
    # Same as lookat, but includes roll
    def lookatBO(self, yaw, pitch, roll, targetPos, lightDirection):
        
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.targetPos = targetPos
        self.lightDirection = lightDirection

        return self.visualize()

    # Resets the debug visualizer camera, and captures an image using the current parameters
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
        return rgbImg, depthImg, segImg

# Class representing camera with adjustable position and orientation
# Very similar to above, main difference is in lookat method, which doesn't include the
# lightDirection parameter.
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

    def visualize(self):

        # self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1, lightPosition=[0, 0, 4])
        self.p.resetDebugVisualizerCamera(self.dis, self.yaw, self.pitch, self.targetPos)




