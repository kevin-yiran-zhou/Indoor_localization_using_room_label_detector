import numpy as np
import cv2

# Function to extract the pose of the label using solvePnP
def pnp(corners, camera_matrix, dist_coeffs, length, height):
    # Define the 3D object points for the room label in the label's coordinate frame
    object_points = np.array([
        [-length/100/2, -height/100/2, 0],  # Bottom-left
        [ length/100/2, -height/100/2, 0],  # Bottom-right
        [ length/100/2,  height/100/2, 0],  # Top-right
        [-length/100/2,  height/100/2, 0]   # Top-left
    ], dtype=np.float32)

    # Convert pixel corner points to numpy array for solvePnP
    if isinstance(corners, list) and len(corners) == 1:
        corners = corners[0]
    corners = [corners[i] for i in [3, 2, 1, 0]]  # Reorder to match object_points
    image_points = np.array(corners, dtype=np.float32)
    
    # Use cv2.solvePnP to calculate rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return None, None, False
    
    return rvec, tvec, True


# Function to convert the rotation vector (rvec) to Euler angles
def rvec_to_euler_angles(rvec):
    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # Extract roll, pitch, and yaw from the rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# Main function to calculate the pose (translation, yaw, pitch, roll) using solvePnP for room labels
def calculate_pose(room_labels_library, detected_label_number, detected_label_corners, camera_matrix, dist_coeffs, scale, resize):
    if len(detected_label_number) != len(detected_label_corners):
        print("Number of detected room labels and corners do not match.")
        return
    if len(detected_label_number) == 0:
        print("No room labels detected.")
        return
    
    for i in range(len(detected_label_number)):
        number = detected_label_number[i]
        for real_room_number in room_labels_library.keys():
            if real_room_number in number:
                break
        break

    print("Final detected room number:", real_room_number)
    corners = detected_label_corners[i]
    print("Detected corners:", corners)
    [label_x, label_y, label_theta, label_length, label_height] = room_labels_library[real_room_number]

    # Calculate translation and rotation using solvePnP
    rvec, tvec, success = pnp(corners, camera_matrix, dist_coeffs, label_length, label_height)
    if not success:
        print(f"Failed to calculate pose for room label '{detected_label_number}'.")
        return None

    # Convert rotation vector to Euler angles (yaw, pitch, roll)
    roll, pitch, yaw = rvec_to_euler_angles(rvec)
    print(f"Roll: {roll:.2f} degrees, Pitch: {pitch:.2f} degrees, Yaw: {yaw:.2f} degrees")

    # Calculate the distance to the label
    print("======================================")
    tvec_resized = tvec * resize
    t_x = tvec_resized[0][0]
    t_y = tvec_resized[1][0]
    t_z = tvec_resized[2][0]
    distance = t_z
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
    distance_in_pixel = int(distance / scale)
    camera_yaw = label_theta - 180 - pitch - horizontal_angle_deg
    camera_x = label_x + distance_in_pixel * (np.sin(np.deg2rad(label_theta)) * np.sin(np.deg2rad(pitch)) + np.cos(np.deg2rad(label_theta)) * np.cos(np.deg2rad(pitch)))
    camera_y = label_y - distance_in_pixel * (np.sin(np.deg2rad(label_theta)) * np.cos(np.deg2rad(pitch)) - np.cos(np.deg2rad(label_theta)) * np.sin(np.deg2rad(pitch)))

    print("======================================")
    print(f"Camera Position: ({camera_x:.2f}, {camera_y:.2f})")
    print(f"Camera Angle: {camera_yaw:.2f} degrees")

    # Return the pose information
    return {
        "x": camera_x,
        "y": camera_y,
        "yaw": camera_yaw
    }