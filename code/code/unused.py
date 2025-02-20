    def runAuto(self, args):

        count = len(self.centerlineArray) - 1

        start_index = len(self.centerlineArray) - 3

        x, y, z = self.centerlineArray[start_index]
        yaw = 0
        pitch = 0
        #pitch, yaw, x, y, z = self.random_start_point(rand_index=start_index) # em teste nao ha random point, comeca no inicio

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

        quat_init = p.getQuaternionFromEuler([pitch, 0, yaw])
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
        direction = np.array([0, 0, 0, 0, 0, 0])  

        while 1:
            tic = time.time()
            p.stepSimulation()

            #0 - Pose atual da camara no mundo
            pitch_current = pitch / 180 * np.pi
            yaw_current = yaw / 180 * np.pi
            quat_current = p.getQuaternionFromEuler([pitch_current, 0, yaw_current])
            R_current = p.getMatrixFromQuaternion(quat_current)
            R_current = np.reshape(R_current, (3, 3))
            T_current = t

            path_trajectoryT.append(t)
            path_trajectoryR.append(p.getMatrixFromQuaternion(quat_current))

            #get Images
            rgb_img, depth_img, rgb_img_ori =  self.get_images(yaw, pitch, t, pos_vector)

            # Smooth the rest centerline and get ground truth camera path from existing path
            nearest_original_centerline_point_sim_cor_index = np.linalg.norm(self.originalCenterlineArray - t, axis=1).argmin()

            if nearest_original_centerline_point_sim_cor_index <= 10:  # reach the target point
                path_centerline_ratio_list.append(1.0)  # complete the path
                break
            else:
                #restSmoothedCenterlineArray = self.smooth_centerline(self.originalCenterlineArray[:m_initalIdx], win_width=10) # filtro media. Apenas ate ponto mais proximo atual, elimina os anteriormente visitados 
                restSmoothedCenterlineArray = self.originalCenterlineArray[:m_initalIdx] # filtro media. Apenas ate ponto mais proximo atual, elimina os anteriormente visitados 
                index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, m_distBO) # vai procurar o ponto do path a seguir, a distancia de m_vel
                index_form_dis2 = self.indexFromDistance(restSmoothedCenterlineArray[:index_form_dis], len(restSmoothedCenterlineArray[:index_form_dis]) - 1, m_distBO) # vai procurar o ponto do path a seguir, a distancia de m_vel
                if not index_form_dis or not index_form_dis2:
                    index_form_dis = len(restSmoothedCenterlineArray) - 2
                    index_form_dis2 = len(restSmoothedCenterlineArray) - 1
                pos_vector_gt = (restSmoothedCenterlineArray[index_form_dis2] - restSmoothedCenterlineArray[index_form_dis]) / np.linalg.norm(restSmoothedCenterlineArray[index_form_dis2] - restSmoothedCenterlineArray[index_form_dis]) # vetor da direcao ideal do movimento

            m_initalIdx = index_form_dis
            pos_vector_norm = np.linalg.norm(pos_vector_gt)
            if pos_vector_norm < 1e-5:
                count -= 1
                continue
        
            # 1 - Proxima posicao camara
            t = restSmoothedCenterlineArray[index_form_dis]

            #2 - calcular qual devera ser a posicao ideal de vista da camara nesse spot
            # para isso ir buscar um ponto a frente x mm e calcular o vector do desejado para esse e definir ai a vista da camara
            pitch = np.arcsin(pos_vector_gt[2] / pos_vector_norm)
            if pos_vector_gt[0] > 0:
                yaw = -np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))  
            else:
                yaw = np.arccos(pos_vector_gt[1] / np.sqrt(pos_vector_gt[0] ** 2 + pos_vector_gt[1] ** 2))
            pitch = pitch / np.pi * 180
            yaw = yaw / np.pi * 180

            # 3 - no mundo da camara, mas nao vai ser usado para ja
            pose_gt_in_camera_cor = np.array([pos_vector_gt[0], -pos_vector_gt[2], pos_vector_gt[1]])
            pitch_gt_in_camera_cor = np.arcsin(-pose_gt_in_camera_cor[1] / np.linalg.norm(pose_gt_in_camera_cor)) #calcula na mesma no frame do mundo
            if pose_gt_in_camera_cor[0] > 0:
                yaw_gt_in_camera_cor = np.arccos(pose_gt_in_camera_cor[2] / np.sqrt(pose_gt_in_camera_cor[0] ** 2 + pose_gt_in_camera_cor[2] ** 2))  
            else:
                yaw_gt_in_camera_cor = -np.arccos(pose_gt_in_camera_cor[2] / np.sqrt(pose_gt_in_camera_cor[0] ** 2 + pose_gt_in_camera_cor[2] ** 2)) #calcula na mesma no frame do mundo
            
            # Caluculate path length
            path_length_diff = np.linalg.norm(t-T_current)
            path_length += path_length_diff
            
            count -= 1
            toc = time.time()
            print("Step frequency:", 1 / (toc - tic))

            rgb_img = np.transpose(rgb_img, axes=(1, 2, 0))
            rgb_img = rgb_img[:, :, ::-1]  # RGB to BGR for showing
            cv2.imshow("saved rgb image", rgb_img)
            cv2.imshow("saved depth image", depth_img)
            cv2.imshow("saved ori image", rgb_img_ori)
            
            cv2.waitKey(5)
        
        p.disconnect()
        self.r.delete()

        return path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, self.originalCenterlineArray, safe_distance_list

        
    def runVS(self, args):
        # Pitch and Yaw
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


            # Manual input to the system 
            keys = p.getKeyboardEvents()
            direction = getAdditionBO(keys, 1)
            direction = np.array(direction)       

            # Get Images from current pose
            rgb_img, depth_img, rgb_img_ori =  self.get_imagesPRY(yaw / np.pi * 180, pitch / np.pi * 180, roll / np.pi * 180, t, pos_vector)

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
            m_jointsVel = [0, 0, 0]

            # The rotational ones will center in the middle, the other will allow to move forward
            m_jointsVel[1] = ((200 - predicted_action_in_image_cor1[0]) - 200 / 2) / (200 / 2)
            m_jointsVel[2] = (predicted_action_in_image_cor1[1] - 200 / 2) / (200 / 2)
            #TEST:
            # A - I only move forward (enable) and the robot control the orientation over time
            m_jointsVel[0] = -direction[2] * 0.005
  
            
            # Implementar IK de acordo com os inputs dados        
            t =  t + np.dot(R_currentCam, [0, 0,  m_jointsVel[0]])  # andar em Z na direcao da camara; OBS: o eixo da camara esta rodado em relacao ao mundo

            quat_step = p.getQuaternionFromEuler([-np.radians(m_jointsVel[2]) * 5, 0, -np.radians(m_jointsVel[1]) * 5]) # pitch 0 yaw
            
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
            path_jointvel.append(m_jointsVel)
            path_joint.append([pitch, roll, yaw])

            cv2.waitKey(5)
        
        p.disconnect()
        self.r.delete()

        return path_trajectoryT, path_trajectoryR, path_centerline_ratio_list, self.originalCenterlineArray, safe_distance_list, path_jointvel, path_joint

"""def getDirectionBO(keys):

    botton_direction = {"＝" : [1, 0, 0, 0, 0, 0],
                        "／" : [0, 1, 0, 0, 0, 0],  
                        "０" : [0, 0, 1, 0, 0, 0],
                        "１" : [0, 0, 0, 1, 0, 0],
                        "２" : [0, 0, 0, 0, 1, 0],
                        "：" : [0, 0, 0, 0, 0, 1]} #enter (65309), #left(65295), right(65296), top(65297), bottom(65298), shift(65305)
    
    for botton in botton_direction.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            print("{} KEY_WAS_TRIGGERED".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            print("{} KEY_IS_DOWN".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            print("{} KEY_WAS_RELEASED".format(botton))
            return botton_direction[botton]
    
    return [0, 0, 0, 0, 0, 0]"""

"""#function to get the direction of the broncoscope
def getDirection(keys):

    botton_direction = {"u" : [1, 0, 0, 0, 0],
                        "h" : [0, 1, 0, 0, 0],  
                        "j" : [0, 0, 1, 0, 0],
                        "k" : [0, 0, 0, 1, 0]}
    
    for botton in botton_direction.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            print("{} KEY_WAS_TRIGGERED".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            print("{} KEY_IS_DOWN".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            print("{} KEY_WAS_RELEASED".format(botton))
            return botton_direction[botton]
    
    return [0, 0, 0, 0, 1]"""

"""def getAddition(keys, scale):
    yaw_add = 0
    pitch_add = 0
    x_add = 0
    y_add = 0
    z_add = 0

    key_actions = {
        "d": ("yaw_add", 1),
        "r": ("pitch_add", 1),
        "k": ("x_add", 1),
        "y": ("y_add", 1),
        "u": ("z_add", 1),
        "f": ("yaw_add", -1),
        "e": ("pitch_add", -1),
        "h": ("x_add", -1),
        "l": ("y_add", -1),
        "j": ("z_add", -1)
    }

    for key, (attr, value) in key_actions.items():
        if ord(key) in keys:
            if keys[ord(key)] & p.KEY_WAS_TRIGGERED:
                locals()[attr] += value
                # print(f"{key} KEY_WAS_TRIGGERED")
            elif keys[ord(key)] & p.KEY_IS_DOWN:
                locals()[attr] += value
                # print(f"{key} KEY_IS_DOWN")
            elif keys[ord(key)] & p.KEY_WAS_RELEASED:
                locals()[attr] += value
                # print(f"{key} KEY_WAS_RELEASED")

    x_add *= scale
    y_add *= scale
    z_add *= scale

    return yaw_add, pitch_add, x_add, y_add, z_add
"""
"""# function to get the addition of the broncoscope
def getAdditionPlain(keys, scale):
    yaw_add = 0
    pitch_add = 0
    x_add = 0
    y_add = 0
    z_add = 0

    key_actions = {
        "u": ("z_add", -1),
        "o": ("z_add", 1),
        "j": ("x_add", -1),
        "l": ("x_add", 1),
        "i": ("y_add", 1),
        "k": ("y_add", -1),
        "f": ("yaw_add", -1),
        "d": ("yaw_add", 1),
        "r": ("pitch_add", 1),
        "e": ("pitch_add", -1)
    }

    for key, (attr, value) in key_actions.items():
        if ord(key) in keys:
            if keys[ord(key)] & p.KEY_WAS_TRIGGERED:
                locals()[attr] += value
                print(f"{key} KEY_WAS_TRIGGERED")
            elif keys[ord(key)] & p.KEY_IS_DOWN:
                locals()[attr] += value
                print(f"{key} KEY_IS_DOWN")
            elif keys[ord(key)] & p.KEY_WAS_RELEASED:
                locals()[attr] += value
                print(f"{key} KEY_WAS_RELEASED")

    x_add *= scale
    y_add *= scale
    z_add *= scale

    return yaw_add, pitch_add, x_add, y_add, z_add"""

# function to get the direction of the broncoscope


"""def getDirectionBO(keys):

    botton_direction = {"＝" : [1, 0, 0, 0, 0, 0],
                        "／" : [0, 1, 0, 0, 0, 0],  
                        "０" : [0, 0, 1, 0, 0, 0],
                        "１" : [0, 0, 0, 1, 0, 0],
                        "２" : [0, 0, 0, 0, 1, 0],
                        "：" : [0, 0, 0, 0, 0, 1]} #enter (65309), #left(65295), right(65296), top(65297), bottom(65298), shift(65305)
    
    for botton in botton_direction.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            print("{} KEY_WAS_TRIGGERED".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            print("{} KEY_IS_DOWN".format(botton))
            return botton_direction[botton]
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            print("{} KEY_WAS_RELEASED".format(botton))
            return botton_direction[botton]
    
    return [0, 0, 0, 0, 0, 0]"""