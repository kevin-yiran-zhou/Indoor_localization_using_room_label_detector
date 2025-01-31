import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# Function to reorder the corners of the detected label
def reorder_corners(corners):
    top_right, top_left, bottom_left, bottom_right = corners
    points = np.array([top_right, top_left, bottom_left, bottom_right])
    y_sorted = points[np.argsort(points[:, 1])]
    top_points = y_sorted[:2]
    bottom_points = y_sorted[2:]
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    return np.array([bottom_left, top_left, bottom_right, top_right], dtype=np.float32)


# Function to find the room label from the detected label number
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
    [label_x, label_y, label_theta, label_length, label_height] = room_labels_library[real_room_number]
    return [label_x, label_y, label_theta, label_length, label_height], reordered_corners



# Main function to calculate the pose (translation, yaw, pitch, roll) using solvePnP for room labels
def calculate_pose(room_labels_library, detected_label_number, detected_label_corners, camera_matrix, dist_coeffs, scale, resize):
    [label_x, label_y, label_theta, label_length, label_height], reordered_corners = find_room(room_labels_library, detected_label_number, detected_label_corners)

    # Calculate translation and rotation using solvePnP
    image_points = reordered_corners
    # Corrected object points (label in XZ-plane) bottom_left, top_left, bottom_right, top_right
    object_points = np.array([
        [-label_length/200, -label_height/200, 0],
        [-label_length/200,  label_height/200, 0],
        [ label_length/200, -label_height/200, 0],
        [ label_length/200,  label_height/200, 0]
    ], dtype=np.float32)
    print("image_points:", image_points)
    print("object_points:", object_points)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        print(f"Failed to calculate pose for room label '{detected_label_number}'.")
        return None

    # Convert the rotation vector to Euler angles
    # https://stackoverflow.com/questions/16265714/camera-pose-estimation-opencv-pnp 
    Rt, _ = cv2.Rodrigues(rvec)
    R = Rt.T
    roll = np.arctan2(-R[2][1], R[2][2])
    pitch = np.arcsin(R[2][0])
    yaw = np.arctan2(-R[1][0], R[0][0])
    print(f"Roll: {np.degrees(roll):.2f} degrees, Pitch: {np.degrees(pitch):.2f} degrees, Yaw: {np.degrees(yaw):.2f} degrees")

    # Calculate the distance to the label
    print(tvec)
    tvec_resized = tvec * resize
    t_x = tvec_resized[0][0]
    t_y = tvec_resized[1][0]
    t_z = tvec_resized[2][0]
    print(f"Translation: ({t_x:.2f}, {t_y:.2f}, {t_z:.2f}) meters")
    print("======================================")
    distance = np.sqrt(t_x**2 + t_z**2) # ignore t_y since it's the height difference
    x_distance = distance * (np.sin(np.deg2rad(label_theta)) * np.sin(pitch) + np.cos(np.deg2rad(label_theta)) * np.cos(pitch))
    y_distance = -distance * (np.sin(np.deg2rad(label_theta)) * np.cos(pitch) - np.cos(np.deg2rad(label_theta)) * np.sin(pitch))
    print(f"Distance: {distance:.2f} meters")
    print(f"X Distance: {x_distance:.2f} meters")
    print(f"Y Distance: {y_distance:.2f} meters")

    # Calculate the angle of the tag from the camera
    horizontal_angle_rad = np.arctan2(t_x, t_z)
    horizontal_angle_deg = np.degrees(horizontal_angle_rad)
    print(f"Horizontal Angle: {horizontal_angle_deg:.2f} degrees")

    # Calculate the camera's position and yaw
    camera_yaw = label_theta - 180 - np.degrees(pitch) - horizontal_angle_deg
    camera_x = label_x + x_distance / scale
    camera_y = label_y + y_distance / scale

    # Return the pose information
    return {
        "x": camera_x,
        "y": camera_y,
        "yaw": camera_yaw
    }