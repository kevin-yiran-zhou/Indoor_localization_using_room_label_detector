import cv2
from room_label_detector import detect_room_label_contours_combined
import json
import os

json_file_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/data/hsv_colors.json"
with open(json_file_path, 'r') as file:
    hsv_colors = json.load(file)
lower_range1 = tuple(hsv_colors["gray1"]["lower"])
upper_range1 = tuple(hsv_colors["gray1"]["upper"])
lower_range2 = tuple(hsv_colors["gray2"]["lower"])
upper_range2 = tuple(hsv_colors["gray2"]["upper"])

folder_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/014-188/origin"
for i in [8]:
    file_name = f"{i}.JPG"
    file_path = os.path.join(folder_path, file_name)
    image = cv2.imread(file_path)
    corners, number = detect_room_label_contours_combined(
        image,
        lower_range2, 
        upper_range2, 
        resize_factor=2, 
        area_threshold=2000, 
        approx_tolerance=0.05, 
        show_result=True
    )

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_dark.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range1, upper_range1, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_dark_different_angle.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range1, upper_range1, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range1, upper_range1, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# # image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_.JPG")
# # corners, number = detect_room_label_contours_combined(image, lower_range1, upper_range1, resize_factor=2.5, area_threshold=5000, approx_tolerance=0.05)

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_different_angle.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range1, upper_range1, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# # image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_different_angle_.JPG")
# # corners, number = detect_room_label_contours_combined(image, lower_range1, upper_range1, resize_factor=2.5, area_threshold=5000, approx_tolerance=0.05)