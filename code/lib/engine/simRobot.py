import numpy as np


# Simulate 2 robots with 3 Dof each
# Robot 1 - Forward/Backward, Pitch, Yaw
# Robot 2 - Forward/Backward, Pitch, Roll

class BroncoRobot1(object):
    def __init__(self, initialPosition = [0, 0, 0], jointlimits = [1000, 170, 120], vellimits = [0.005, 100, 5], accelaration = [100, 10, 10]):
        self.jointvalues = initialPosition # initial configuration of the robot
        self.joint_limits = jointlimits # mm, degree, degree
        self.vel_limits = vellimits # mm, deg/s, deg/s
        self.accelaration = accelaration # mm/s2, deg/s2, deg/s2
    
    def setJoints(self, jointvalues):
        self.jointvalues = jointvalues
    
    def getJoints(self):
        return self.jointvalues
    
    def visualservoingcontrol(self, imgTarget, imgSize = [200, 200], theta1_range = 20, theta2_range = 25):
        
        m_middleImage = np.divide(imgSize,2)
        m_jointsVelRel = [0, 0, 0]

        m_currJoints = self.getJoints()

        #Working (arc length from current position to the desired one) Normalize between [0 : 1], 1 corrspondes to a angle diff of 90 in the limit of the image (pi/2 * raio(100 pixels))
        m_jointsVelRel[1] = (np.arcsin((imgTarget[0] - m_middleImage[0])/(np.linalg.norm(imgTarget[:-1] - m_middleImage))) * np.linalg.norm(imgTarget[:-1] - m_middleImage)) \
        / (np.pi / 2 * 100) # calcula o perimetro que falta percorrer para alinhar com o eixo. quanto menor o perimetro, menor a velocidade


        # Nao esta a entrar aqui
        if((imgTarget[0] <= 0 and imgTarget[1] >= 0) or \
            (imgTarget[0] > 0 and imgTarget[1] < 0)):
            m_jointsVelRel[1] = - (m_jointsVelRel[1])

        # Sequential control, define limits where each joint will work

        if(np.linalg.norm(imgTarget[:-1] - m_middleImage[0]) < 20 or np.abs(m_jointsVelRel[1]) < 0.1):
                m_jointsVelRel[1] = 0
                m_jointsVelRel[2] = (imgTarget[1] - m_middleImage[1]) / (m_middleImage[1])
                

        # Compute next joint values
        m_nextJoints = np.add(m_currJoints, np.multiply(-np.radians(m_jointsVelRel), self.vel_limits))

        # Joint lmits
        
        # Theta 1 - the rotation of the bronchoscope
        # if the next joint passes a limit, the robot rotate around to get the desired value to the joint
        if(np.abs(m_nextJoints[1]) > np.radians(self.joint_limits[1])):
            m_nextJoints[2] = m_currJoints[2] # nao rodar a junta2, apenas roda a 3 para sair do limite de junta

            if(m_nextJoints[1] > np.radians(self.joint_limits[1])):
                m_nextJoints[1] = np.minimum(np.maximum(m_currJoints[1] + (- np.radians(180) + (m_jointsVelRel[1] * self.vel_limits[1])), np.radians(-self.joint_limits[1])), np.radians(self.joint_limits[1]))
            else:
                m_nextJoints[1] = np.minimum(np.maximum(m_currJoints[1] + (np.radians(180) + (m_jointsVelRel[1] * self.vel_limits[1])), np.radians(-self.joint_limits[1])), np.radians(self.joint_limits[1]))

        # Check if theta 2 is ok
        else:
            # Theta 2 - rotate the theta 1 do change the direction of movement of theta 2
            if(np.abs(m_nextJoints[2]) > np.radians(self.joint_limits[2])):
                m_nextJoints[2] = m_currJoints[2] # nao rodar a junta2, apenas roda a 3 para sair do limite de junta
                if(m_nextJoints[1] >= 0):
                    m_nextJoints[1] = np.maximum(m_nextJoints[1] - np.radians(180), - np.radians(self.joint_limits[1]))
                else:
                    m_nextJoints[1] = np.minimum(m_nextJoints[1] + np.radians(180), np.radians(self.joint_limits[1]))
        
        self.setJoints(m_nextJoints)
        
        return m_jointsVelRel,m_nextJoints

    def positionservoingcontrol(self, targetpose, imgSize = [200, 200]):

        # From the current position, compute the movement to the next position
        m_joints = [0, 0, 0]

        return m_joints



