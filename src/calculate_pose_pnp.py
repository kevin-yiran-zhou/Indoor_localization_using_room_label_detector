import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def reorder_corners(corners):
    top_right, top_left, bottom_left, bottom_right = corners
    points = np.array([top_right, top_left, bottom_left, bottom_right])
    y_sorted = points[np.argsort(points[:, 1])]
    top_points = y_sorted[:2]
    bottom_points = y_sorted[2:]
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    return np.array([top_right, top_left, bottom_left, bottom_right], dtype=np.float32)


def find_room(room_labels_library, detected_label_number, detected_label_corners):
    if len(detected_label_number) != len(detected_label_corners):
        print("Number of detected room labels and corners do not match.")
        return
    if len(detected_label_number) == 0:
        print("No room labels detected.")
        return
    
    target_room_number = None
    finished = False
    i = 0
    while not finished and i < len(detected_label_number):
        number = detected_label_number[i]
        for real_room_number in room_labels_library.keys():
            if real_room_number in number:
                finished = True
                target_room_number = real_room_number
                break
        i += 1
    if not finished:
        print("No matching room label found.")
        return

    print("Final detected room number:", target_room_number)
    corners = detected_label_corners[i - 1]
    reordered_corners = reorder_corners(corners)
    print("Detected corners:", reordered_corners)
    [label_x, label_y, label_theta, label_length, label_height] = room_labels_library[real_room_number]
    return [label_x, label_y, label_theta, label_length, label_height], reordered_corners


# Function to extract the pose of the label using solvePnP
def pnp(reordered_corners, camera_matrix, dist_coeffs, length, height):
    image_points = reordered_corners

    # Corrected object points (label in XZ-plane)
    # top_right, top_left, bottom_left, bottom_right
    object_points = np.array([
        [ length/200,  height/200, 0],
        [-length/200,  height/200, 0],
        [-length/200, -height/200, 0],
        [ length/200, -height/200, 0]
    ], dtype=np.float32)
    
    # Use cv2.solvePnP to calculate rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return None, None, False
    
    return rvec, tvec, True


# # Function to convert the rotation vector (rvec) to Euler angles
# def rvec_to_euler_angles(rvec):
#     # Convert the rotation vector to a rotation matrix
#     rotation_matrix, _ = cv2.Rodrigues(rvec)
#     print("Rotation Matrix:")
#     print(rotation_matrix)
    
#     # Extract roll, pitch, and yaw from the rotation matrix
#     sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
#     singular = sy < 1e-6

#     if not singular:
#         roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
#         pitch = np.arctan2(-rotation_matrix[2, 0], sy)
#         yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
#     else:
#         roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
#         pitch = np.arctan2(-rotation_matrix[2, 0], sy)
#         yaw = 0

#     return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# Main function to calculate the pose (translation, yaw, pitch, roll) using solvePnP for room labels
def calculate_pose(room_labels_library, detected_label_number, detected_label_corners, camera_matrix, dist_coeffs, scale, resize):
    [label_x, label_y, label_theta, label_length, label_height], reordered_corners = find_room(room_labels_library, detected_label_number, detected_label_corners)

    # Calculate translation and rotation using solvePnP
    rvec, tvec, success = pnp(reordered_corners, camera_matrix, dist_coeffs, label_length, label_height)
    if not success:
        print(f"Failed to calculate pose for room label '{detected_label_number}'.")
        return None

    # Convert rotation vector to Euler angles (yaw, pitch, roll)
    Rt, _ = cv2.Rodrigues(rvec)
    R = Rt.transpose()
    roll = np.arctan2(-R[2][1], R[2][2])
    pitch = np.arcsin(R[2][0])
    yaw = np.arctan2(-R[1][0], R[0][0])
    print(f"Roll: {np.degrees(roll):.2f} degrees, Pitch: {np.degrees(pitch):.2f} degrees, Yaw: {np.degrees(yaw):.2f} degrees")

    # Calculate the distance to the label
    tvec_resized = tvec * resize
    t_x = tvec_resized[0][0]
    t_y = tvec_resized[1][0]
    t_z = tvec_resized[2][0]
    print(f"Translation: ({t_x:.2f}, {t_y:.2f}, {t_z:.2f}) meters")
    print("======================================")
    distance = np.sqrt(t_x**2 + t_z**2) # ignore t_y since it's the height difference
    x_distance = distance * (np.sin(np.deg2rad(label_theta)) * np.sin(np.deg2rad(pitch)) + np.cos(np.deg2rad(label_theta)) * np.cos(np.deg2rad(pitch)))
    y_distance = -distance * (np.sin(np.deg2rad(label_theta)) * np.cos(np.deg2rad(pitch)) - np.cos(np.deg2rad(label_theta)) * np.sin(np.deg2rad(pitch)))
    print(f"Distance: {distance:.2f} meters")
    print(f"X Distance: {x_distance:.2f} meters")
    print(f"Y Distance: {y_distance:.2f} meters")

    # Calculate the angle of the tag from the camera
    horizontal_angle_rad = np.arctan2(t_x, t_z)
    horizontal_angle_deg = np.degrees(horizontal_angle_rad)
    print(f"Horizontal Angle: {horizontal_angle_deg:.2f} degrees")

    # Calculate the camera's position and yaw
    camera_yaw = label_theta - 180 - pitch - horizontal_angle_deg
    camera_x = label_x + x_distance / scale
    camera_y = label_y + y_distance / scale

    # Return the pose information
    return {
        "x": camera_x,
        "y": camera_y,
        "yaw": camera_yaw
    }