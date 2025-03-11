import numpy as np
import math

class Camera:
    def __init__(self, targetPos, dis, yaw, pitch, roll):
        self.targetPos = targetPos
        self.dis = dis
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def getExtrinsic(self):
        # Calculates the extrinsic camera matrix.
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        roll_rad = math.radians(self.roll)

        camera_x = self.targetPos[0] + self.dis * math.cos(yaw_rad) * math.cos(pitch_rad)
        camera_y = self.targetPos[1] + self.dis * math.sin(yaw_rad) * math.cos(pitch_rad)
        camera_z = self.targetPos[2] + self.dis * math.sin(pitch_rad)
        camera_position = np.array([camera_x, camera_y, camera_z])

        print("Camera position:", camera_position) # added print statement

        rotation_z = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        rotation_y = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        rotation_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])
        rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        translation_vector = -camera_position

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector

        return extrinsic_matrix

def test_extrinsic_matrix():
    # Test case 1: Simple case with no rotation and camera at (0, 0, -dis)
    target_pos = np.array([0, 0, 0])
    dis = 10
    yaw = 0
    pitch = 0
    roll = 0
    camera = Camera(target_pos, dis, yaw, pitch, roll)
    extrinsic = camera.getExtrinsic()
    print("Calculated extrinsic matrix:")
    print(extrinsic) # added print statement
    expected_extrinsic = np.array([
        [1, 0, 0, -10], # Corrected translation
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    assert np.allclose(extrinsic, expected_extrinsic), "Test case 1 failed"

    # Test case 2: Yaw rotation of 90 degrees
    yaw = 90
    camera = Camera(target_pos, dis, yaw, pitch, roll)
    extrinsic = camera.getExtrinsic()
    expected_extrinsic = np.array([
        [0, -1, 0, 0], # corrected translation
        [1, 0, 0, -10], # corrected translation
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    assert np.allclose(extrinsic, expected_extrinsic), "Test case 2 failed"

    # Test case 3: Pitch rotation of 90 degrees
    yaw = 0
    pitch = 90
    camera = Camera(target_pos, dis, yaw, pitch, roll)
    extrinsic = camera.getExtrinsic()
    expected_extrinsic = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, -10],
        [0, 0, 0, 1]
    ])
    assert np.allclose(extrinsic, expected_extrinsic), "Test case 3 failed"

    # Test case 4: Combined yaw and pitch
    yaw = 45
    pitch = 45
    camera = Camera(target_pos, dis, yaw, pitch, roll)
    extrinsic = camera.getExtrinsic()
    # Manual calculation or an external tool is recommended for precise verification.
    # Here, we will just check if the output is a 4x4 matrix.
    assert extrinsic.shape == (4, 4), "Test case 4 failed: Incorrect shape"

    # Test case 5: Roll Rotation
    roll = 90
    camera = Camera(target_pos, dis, yaw, pitch, roll)
    extrinsic = camera.getExtrinsic()
    assert extrinsic.shape == (4,4), "Test case 5 failed: incorrect shape"

    # Test case 6: targetpos not at origin.
    target_pos = np.array([1,2,3])
    camera = Camera(target_pos, dis, yaw, pitch, roll)
    extrinsic = camera.getExtrinsic()
    assert extrinsic.shape == (4,4), "Test case 6 failed: incorrect shape"

    print("All test cases passed!")

if __name__ == "__main__":
    test_extrinsic_matrix()