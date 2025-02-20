import serial
import numpy as np
import time

class brochoRobClass(object):

    def __init__(self, usbPort = 'COM7', baundrate = 115200, timestep = 0.05, jointlimits = [170,170,250], vellimits = [25,25,100]):

        self.ser = serial.Serial(usbPort, baundrate, timeout=1)
        time.sleep(0.5)

        # initialization - home position
        self.ser.write(b'i\n')
        time.sleep(0.5)

        self.jointvalues = []
        
        self.timestep = timestep

        self.joint_limits = jointlimits # mm, degree, degree
        self.vel_limits = vellimits # deg/s, deg/s, mm
        #self.accelaration = accelaration # mm/s2, deg/s2, deg/s2

        self.jointBendingConvert = [850,1700] # valores pwd, representa [-170,170]
        self.jointRotationConvert = [544,2400] # valores pwd, representa [-170,170]
        self.jointTranslationConvert = [0,3600*64] # valores step, 250 mm //AQUI 64, o valor do microstepper que estamos a usar agora

        self.getJoints() #[Bending, Rotation, Translation]

        self.jointvelControl = 0.25

        self.m_joystick = 0

    # Read current value of joints
    def getJoints(self):
        self.ser.write(b'j\n')
        self.jointvalues = []
        while True: # READ initial robot position (we should add a time limit here)
            if self.ser.in_waiting > 0:
                data = self.ser.readline().decode('utf-8').strip()
                #self.jointvalues.append(int(data))
                self.jointvalues.append(((int(data) - self.jointBendingConvert[0]) / (self.jointBendingConvert[1] - self.jointBendingConvert[0])) * (self.joint_limits[0]*2) - self.joint_limits[0]) #((value-min) / (max-min) * jointrange) - halfjointrange
                data = self.ser.readline().decode('utf-8').strip()
                #self.jointvalues.append(int(data))
                self.jointvalues.append(((int(data) - self.jointRotationConvert[0]) / (self.jointRotationConvert[1] - self.jointRotationConvert[0])) * (self.joint_limits[1]*2) - self.joint_limits[1]) #((value-min) / (max-min) * jointrange) - halfjointrange
                data = self.ser.readline().decode('utf-8').strip()
                #self.jointvalues.append(int(data))
                self.jointvalues.append(((int(data) - self.jointTranslationConvert[0]) / (self.jointTranslationConvert[1] - self.jointTranslationConvert[0])) * self.joint_limits[2]) #((value-min) / (max-min) * jointrange) - halfjointrange
                break
        return self.jointvalues.copy()

    # Set the joint positions. At this stage, the arduino side is controlling the joint value incrementally (+ 1)
    def setJointsIncremental(self, destinationJoints):
        try:
            # 1 - Bending Joint
            if (destinationJoints[0] != -1) :
                if(destinationJoints[0] == 1):
                    self.ser.write(b'u\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[0] = int(data)
                            break
                    time.sleep(self.timestep)
                elif destinationJoints[0] == -1:
                    self.ser.write(b'd\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[0] = int(data)
                            break
                    time.sleep(self.timestep)

            # 2 - Rotation joint
            if(destinationJoints[1] != 0): 
                if(destinationJoints[1] == 1):
                    self.ser.write(b'l\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[1] = int(data)
                            break
                    time.sleep(self.timestep)
                elif(destinationJoints[1] == -1):
                    self.ser.write(b'r\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[1] = int(data)
                            break
                    time.sleep(self.timestep)

            #3 - Translational joint
            if(destinationJoints[2] != 0): #rotation joint
                if(destinationJoints[2] == 1):
                    self.ser.write(b'f\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[2] = int(data)
                            break
                    time.sleep(self.timestep)
                elif(destinationJoints[2] == -1):
                    self.ser.write(b'b\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[2] = int(data)
                            break
                    time.sleep(self.timestep)
            return True
        except Exception as e:
            raise e

    # Set the joint positions, defining the desired position
    def setJoints(self, destinationJoints):
        try:
            # 1 - Bending Joint
            if(destinationJoints[0]  != self.jointvalues[0]):

                # Define vel limits security
                #m_step = min(abs(destinationJoints[0]-self.jointvalues[0]), self.vel_limits[0])
                #m_sign =  np.sign(destinationJoints[0]-self.jointvalues[0])
                #destinationJoints[0] = self.jointvalues[0] + m_sign * m_step

                cmd = 'b' + str(int(((destinationJoints[0] + self.joint_limits[0]) / (self.joint_limits[0] * 2)) * (self.jointBendingConvert[1] - self.jointBendingConvert[0]) + self.jointBendingConvert[0])) + '\n'
                #cmd = 'b' + str(destinationJoints[0]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[0] = ((int(data) - self.jointBendingConvert[0]) / (self.jointBendingConvert[1] - self.jointBendingConvert[0])) * (self.joint_limits[0]*2) - self.joint_limits[0]
                        break
                #time.sleep(self.timestep)

            # 2 - Rotation joint
            if(destinationJoints[1]  != self.jointvalues[1]): 

                # Define vel limits security
                #m_step = min(abs(destinationJoints[1] - self.jointvalues[1]), self.vel_limits[1])
                #m_sign =  np.sign(destinationJoints[1] - self.jointvalues[1])
                #destinationJoints[1] = self.jointvalues[1] + m_sign * m_step


                cmd = 'r' + str(int(((destinationJoints[1] + self.joint_limits[1]) / (self.joint_limits[1] * 2)) * (self.jointRotationConvert[1] - self.jointRotationConvert[0]) + self.jointRotationConvert[0])) + '\n'
                #cmd = 'r' + str(destinationJoints[1]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[1] = ((int(data) - self.jointRotationConvert[0]) / (self.jointRotationConvert[1] - self.jointRotationConvert[0])) * (self.joint_limits[1]*2) - self.joint_limits[1]
                        break
                #time.sleep(self.timestep)

            #3 - Translational joint
            if(destinationJoints[2] != self.jointvalues[2]): #rotation joint

                # Define vel limits security
                #m_step = min(abs(destinationJoints[2] - self.jointvalues[2]), self.vel_limits[2])
                #m_sign =  np.sign(destinationJoints[2] - self.jointvalues[2])
                #destinationJoints[2] = self.jointvalues[2] + m_sign * m_step
                m_initialvalue = self.jointvalues[2]

                if (destinationJoints[2] < 0):
                    destinationJoints[2] = 0

                cmd = 't' + str(int(((destinationJoints[2]) / (self.joint_limits[2])) * (self.jointTranslationConvert[1] - self.jointTranslationConvert[0]) + self.jointTranslationConvert[0])) + '\n'
                #cmd = 't' + str(destinationJoints[2]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo                
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[2] = round(((int(data) - self.jointTranslationConvert[0]) / (self.jointTranslationConvert[1] - self.jointTranslationConvert[0])) * self.joint_limits[2],1)
                        break
                #time.sleep(self.timestep)
                    
            return True
        except Exception as e:
            raise e
        


        # Set the joint positions, defining the desired position
    
    # To control the robot with the joystick in a continuous way
    def setJointsTest(self, destinationJoints):
        try:
            # 1 - Bending Joint
            if(destinationJoints[0]  != self.jointvalues[0]):
                #start = time.time()                 
                # Define vel limits security
                #m_step = min(abs(destinationJoints[0]-self.jointvalues[0]), self.vel_limits[0])
                #m_sign =  np.sign(destinationJoints[0]-self.jointvalues[0])
                #destinationJoints[0] = self.jointvalues[0] + m_sign * m_step

                cmd = 'b' + str(int(((destinationJoints[0] + self.joint_limits[0]) / (self.joint_limits[0] * 2)) * (self.jointBendingConvert[1] - self.jointBendingConvert[0]) + self.jointBendingConvert[0])) + '\n'
                #cmd = 'b' + str(destinationJoints[0]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[0] = ((int(data) - self.jointBendingConvert[0]) / (self.jointBendingConvert[1] - self.jointBendingConvert[0])) * (self.joint_limits[0]*2) - self.joint_limits[0]
                        break
                #time.sleep(self.timestep)
                #end = time.time()
                #print(end - start)
            # 2 - Rotation joint
            if(destinationJoints[1]  != self.jointvalues[1]): 
                #start = time.time()
                # Define vel limits security
                #m_step = min(abs(destinationJoints[1] - self.jointvalues[1]), self.vel_limits[1])
                #m_sign =  np.sign(destinationJoints[1] - self.jointvalues[1])
                #destinationJoints[1] = self.jointvalues[1] + m_sign * m_step


                cmd = 'r' + str(int(((destinationJoints[1] + self.joint_limits[1]) / (self.joint_limits[1] * 2)) * (self.jointRotationConvert[1] - self.jointRotationConvert[0]) + self.jointRotationConvert[0])) + '\n'
                #cmd = 'r' + str(destinationJoints[1]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[1] = ((int(data) - self.jointRotationConvert[0]) / (self.jointRotationConvert[1] - self.jointRotationConvert[0])) * (self.joint_limits[1]*2) - self.joint_limits[1]
                        break

                #end = time.time()
                #print(end - start)
                #time.sleep(self.timestep)

            #3 - Translational joint
            if(destinationJoints[2] != self.m_joystick): #rotation joint
                #start = time.time()

                if (destinationJoints[2] == -1):
                    cmd = 'b' + '\n'
                    self.m_joystick = -1
                elif (destinationJoints[2] == 1):
                    cmd = 'f' + '\n'
                    self.m_joystick = 1
                else:
                    cmd = 's' + '\n'
                    self.m_joystick = 0
                    
                self.ser.write(cmd.encode()) # send desired joint value to the servo         
                
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[2] = round(((int(data) - self.jointTranslationConvert[0]) / (self.jointTranslationConvert[1] - self.jointTranslationConvert[0])) * self.joint_limits[2],1)
                        break
                #time.sleep(self.timestep)
                #end = time.time()
                #print(end - start)
                    
            return True
        except Exception as e:
            raise e

    # Visual servoing control of the robot based on the desired image size
    def visualservoingcontrol(self, imgTarget, m_transStep, imgSize = [400, 400]):
        
        m_middleImage = np.divide(imgSize,2)
        m_jointsVelRel = [0, 0, 0]

        m_currJoints = self.getJoints()

        #Working (arc length from current position to the desired one) Normalize between [0 : 1], 1 corrspondes to a angle diff of 90 in the limit of the image (pi/2 * raio(100 pixels))
        m_jointsVelRel[1] = (np.arcsin((imgTarget[0] - m_middleImage[0])/(np.linalg.norm(imgTarget - m_middleImage))) * np.linalg.norm(imgTarget - m_middleImage)) \
        / (np.pi / 2 * 200) # calcula o perimetro que falta percorrer para alinhar com o eixo. quanto menor o perimetro, menor a velocidade


        # Nao esta a entrar aqui
        if((imgTarget[0] <= 0 and imgTarget[1] >= 0) or \
            (imgTarget[0] > 0 and imgTarget[1] < 0)):
            m_jointsVelRel[1] = - (m_jointsVelRel[1])

        # Sequential control, define limits where each joint will work

        if(np.linalg.norm(imgTarget - m_middleImage) < 20 or np.abs(m_jointsVelRel[1]) < 0.1):
                m_jointsVelRel[1] = 0
                m_jointsVelRel[0] = (imgTarget[1] - m_middleImage[1]) / (m_middleImage[1])
                

        # Compute next joint values
        m_nextJoints = []
        for i in range(3):
            m_nextJoints.append(m_currJoints[i] + m_jointsVelRel[i] * self.vel_limits[i] * self.jointvelControl)

        # to test
        m_jointsVelRel[2] = m_transStep * 0.005
        m_nextJoints[2] = m_transStep * 0.005

        # Joint lmits
        '''
        # Theta 1 - the rotation of the bronchoscope
        # if the next joint passes a limit, the robot rotate around to get the desired value to the joint
        if(np.abs(m_nextJoints[0]) > np.radians(self.joint_limits[0])):
            m_nextJoints[1] = m_currJoints[1] # nao rodar a junta2, apenas roda a 3 para sair do limite de junta

            if(m_nextJoints[0] > np.radians(self.joint_limits[0])):
                m_nextJoints[0] = np.minimum(np.maximum(m_currJoints[0] + (- np.radians(180) + (m_jointsVelRel[0] * self.vel_limits[0])), np.radians(-self.joint_limits[0])), np.radians(self.joint_limits[0]))
            else:
                m_nextJoints[0] = np.minimum(np.maximum(m_currJoints[0] + (np.radians(180) + (m_jointsVelRel[0] * self.vel_limits[0])), np.radians(-self.joint_limits[0])), np.radians(self.joint_limits[0]))

        # Check if theta 2 is ok
        else:
            # Theta 2 - rotate the theta 1 do change the direction of movement of theta 2
            if(np.abs(m_nextJoints[1]) > np.radians(self.joint_limits[1])):
                m_nextJoints[1] = m_currJoints[1] # nao rodar a junta2, apenas roda a 3 para sair do limite de junta
                if(m_nextJoints[0] >= 0):
                    m_nextJoints[0] = np.maximum(m_nextJoints[0] - np.radians(180), - np.radians(self.joint_limits[0]))
                else:
                    m_nextJoints[0] = np.minimum(m_nextJoints[0] + np.radians(180), np.radians(self.joint_limits[0]))'''
        
        ##self.setJoints(m_nextJoints)
        
        return m_jointsVelRel,m_nextJoints