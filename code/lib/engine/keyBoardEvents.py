
import pybullet as p


def getDirectionBO(keys):

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
    
    return [0, 0, 0, 0, 0, 0]


def getAdditionBO(keys, scale):

    yaw_add = 0
    pitch_add = 0
    z_add = 0

    botton_add_dict = {"／" : yaw_add,
                        "１" : pitch_add,  
                        "＝" : z_add}
    botton_minus_dict = {"０" : yaw_add,
                            "２" : pitch_add,
                            "：" : z_add}
    
    for botton in botton_add_dict.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            botton_add_dict[botton] += 1
            # print("{} KEY_WAS_TRIGGERED".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            botton_add_dict[botton] += 1
            # print("{} KEY_IS_DOWN".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            botton_add_dict[botton] += 1
            # print("{} KEY_WAS_RELEASED".format(botton))

    for botton in botton_minus_dict.keys():
        if ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_TRIGGERED:
            botton_minus_dict[botton] -= 1
            # print("{} KEY_WAS_TRIGGERED".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_IS_DOWN:
            botton_minus_dict[botton] -= 1
            # print("{} KEY_IS_DOWN".format(botton))
        elif ord(botton) in keys and keys[ord(botton)] & p.KEY_WAS_RELEASED:
            botton_minus_dict[botton] -= 1
            # print("{} KEY_WAS_RELEASED".format(botton))

    yaw_add = botton_add_dict["／"] + botton_minus_dict["０"]
    pitch_add = botton_add_dict["１"] + botton_minus_dict["２"]
    z_add = (botton_add_dict["＝"] + botton_minus_dict["："]) * scale
    
    return yaw_add, pitch_add, z_add


