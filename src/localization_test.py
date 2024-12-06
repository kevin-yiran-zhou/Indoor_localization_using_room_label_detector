from room_label_detector import detect_room_label_contours_combined
from calculate_pose_pnp import calculate_pose
import cv2
import json
import os
import numpy as np


json_file_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/data/hsv_colors.json"
with open(json_file_path, 'r') as file:
    hsv_colors = json.load(file)
lower_range = tuple(hsv_colors["gray"]["lower"])
upper_range = tuple(hsv_colors["gray"]["upper"])


org_image_width = 4031
org_image_height = 3023
resize_factor = 4
resize = 1/resize_factor
image_width = int(org_image_width * resize)
image_height = int(org_image_height * resize)
camera_focal_length = 26 * image_width / 7.03
c_x = round(image_width / 2)
c_y = round(image_height / 2)
camera_matrix = np.array([[camera_focal_length, 0, c_x],
                        [0, camera_focal_length, c_y],
                        [0, 0, 1]])
dist_coeffs = np.zeros((1, 5))


# Test the function
image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_dark.JPG")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=False)

floor_data_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/data/maps/basic-floor-plan.json"
with open(floor_data_path, 'r') as file:
    floor_data = json.load(file)
    room_labels_library = {
        name: [room[0], room[1], room[2], room[3], room[4]]
        for name, room in floor_data.get("rooms", {}).items()
    }
    scale = floor_data.get("scale", 1)
print("scale:", scale)
print("Available Room Numbers:", room_labels_library.keys())

print("====================================")
pose = calculate_pose(room_labels_library, number, corners, camera_matrix, dist_coeffs, scale, resize)
if pose:
    x = pose["x"]
    y = pose["y"]
    theta = pose["yaw"]
    print("Camera Position (pixel on floorplan):", x, y)
    print("Camera Orientation (degrees):", theta)
else:
    print("Localization failed.")