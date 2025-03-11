import numpy as np
import math

def extract_orientation(camera_data):
    orientation_log = open("camera_orientation_log.txt", "w")
    orientation_log.write("Timestep, Roll, Pitch, Yaw\n")

    translation_rotation_log = open("translation_rotation_log.txt", "w")
    translation_rotation_log.write("Timestep, Translation, Rotation\n")

    for timestep, data in enumerate(camera_data):
        rotation_matrix = data['rotation_matrix']
        translation_vector = data['translation_vector']

        # Calculate roll, pitch, yaw from rotation matrix
        roll = math.atan2(rotation_matrix[2][1], rotation_matrix[2][2])
        pitch = math.atan2(-rotation_matrix[2][0], math.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2))
        yaw = math.atan2(rotation_matrix[1][0], rotation_matrix[0][0])

        orientation_log.write(f"{timestep}, {roll}, {pitch}, {yaw}\n")

        if timestep % 100 == 0:
            translation_rotation_log.write(f"{timestep}, {translation_vector.tolist()}, {rotation_matrix.tolist()}\n")

    orientation_log.close()
    translation_rotation_log.close()

def main():
    # Example camera data
    camera_data = []
    for i in range(1000):  # Generate 1000 timesteps of data
        rotation_matrix = np.array([
            [np.cos(i * 0.01), -np.sin(i * 0.01), 0],
            [np.sin(i * 0.01), np.cos(i * 0.01), 0],
            [0, 0, 1]
        ])
        translation_vector = np.array([i * 0.1, i * 0.1, i * 0.1])
        camera_data.append({'rotation_matrix': rotation_matrix, 'translation_vector': translation_vector})
    
    extract_orientation(camera_data)

if __name__ == "__main__":
    main()
