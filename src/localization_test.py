from room_label_detector import detect_room_label_contours_combined
from calculate_pose_pnp import calculate_pose
import cv2
import json
import os
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


json_file_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/data/hsv_colors.json"
with open(json_file_path, 'r') as file:
    hsv_colors = json.load(file)
lower_range1 = tuple(hsv_colors["gray1"]["lower"])
upper_range1 = tuple(hsv_colors["gray1"]["upper"])
lower_range2 = tuple(hsv_colors["gray2"]["lower"])
upper_range2 = tuple(hsv_colors["gray2"]["upper"])


# iPhone 12 Pro Max camera parameters
# org_image_width = 4031
# org_image_height = 3023
# resize_factor = 4
# resize = 1/resize_factor
# image_width = int(org_image_width * resize)
# image_height = int(org_image_height * resize)
# # camera_focal_length = Equivalent focal length * Sensor width / Image width
# # Main camera focal length: 26mm, sensor width: 7.03mm
# camera_focal_length = 26 * image_width / 7.03
# c_x = round(image_width / 2)
# c_y = round(image_height / 2)
# camera_matrix = np.array([[camera_focal_length, 0, c_x],
#                         [0, camera_focal_length, c_y],
#                         [0, 0, 1]])
# dist_coeffs = np.zeros((1, 5))

# iPhone 16 Max camera parameters from EXIF
sensor_width = 5.96  # mm
focal_length_mm = 5.96  # mm
org_image_width = 5712
org_image_height = 4284
resize_factor = 2
resize = 1 / resize_factor
image_width = int(org_image_width * resize)
image_height = int(org_image_height * resize)
# Compute original focal length in pixels (before resizing)
camera_focal_length_px = (focal_length_mm * org_image_width) / sensor_width  
# Adjust focal length for resized image
camera_focal_length_px_resized = camera_focal_length_px / resize_factor  
# Principal point (center of resized image)
c_x = round(image_width / 2)
c_y = round(image_height / 2)
# Corrected Intrinsic Camera Matrix
camera_matrix = np.array([
    [camera_focal_length_px_resized, 0, c_x],
    [0, camera_focal_length_px_resized, c_y],
    [0, 0, 1]
])
# Assume no lens distortion
dist_coeffs = np.zeros((1, 5))


# Test the function
path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/014-188/origin/2.JPG"
print("Processing image:", path)

###
# img = Image.open(path)
# exif_data = img._getexif()
# for tag, value in exif_data.items():
#     tag_name = TAGS.get(tag, tag)
#     print(f"{tag_name}: {value}")
###

image = cv2.imread(path)
corners, number = detect_room_label_contours_combined(image, lower_range2, upper_range2, resize_factor=resize_factor, area_threshold=2000, approx_tolerance=0.05, show_result=False)

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