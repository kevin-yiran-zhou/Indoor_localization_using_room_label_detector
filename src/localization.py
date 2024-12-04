import cv2
from room_label_detector import detect_room_label_contours_combined
import json
import os
import numpy as np

def localization_with_room_label(corners, number, room_labels_library, camera_matrix, dist_coeffs, scale):
    if len(number) != len(corners):
        print("Number of detected room labels and corners do not match.")
        return
    if len(number) == 0:
        print("No room labels detected.")
        return
    
    for i in range(len(number)):
        detected_room_number = number[i]
        for room_number in room_labels_library.keys():
            if room_number in detected_room_number:
                break
        break

    print("Final detected room number:", room_number)
    [x_label, y_label, theta_label, length, height] = room_labels_library[room_number]

    # Define 3D object points of the room label (assume it's centered at [0, 0, 0])
    object_points = np.array([
        [-length / 2, -height / 2, 0],  # Bottom-left
        [ length / 2, -height / 2, 0],  # Bottom-right
        [ length / 2,  height / 2, 0],  # Top-right
        [-length / 2,  height / 2, 0]   # Top-left
    ], dtype=np.float32)

    # Get the corresponding image points (corners in pixel coordinates)
    image_points = np.array(corners[i], dtype=np.float32)

    # Use solvePnP to calculate the rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        print("Pose calculation failed for room label.")
        return None

    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Extract yaw angle from the rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Yaw in radians

    # Convert translation vector (tvec) to global coordinates
    camera_x = x_label + (tvec[0][0] * scale)
    camera_y = y_label + (tvec[1][0] * scale)
    camera_yaw = np.degrees(yaw) + theta_label

    return [camera_x, camera_y, camera_yaw]