
import pybullet as p

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

#function to get the addition of the broncoscope
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
