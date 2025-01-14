import cv2
from room_label_detector import detect_room_label_contours_combined
import json
import os

json_file_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/data/hsv_colors.json"
with open(json_file_path, 'r') as file:
    hsv_colors = json.load(file)
lower_range = tuple(hsv_colors["gray"]["lower"])
upper_range = tuple(hsv_colors["gray"]["upper"])

image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/test/resized/1_30.jpg")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/test/resized/1_60.jpg")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/test/resized/2_30.jpg")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/test/resized/2_60.jpg")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/test/resized/3_30.jpg")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/test/resized/3_60.jpg")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/test/resized/3_90.jpg")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_dark.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_dark_different_angle.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# # image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_.JPG")
# # corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=2.5, area_threshold=5000, approx_tolerance=0.05)

# image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_different_angle.JPG")
# corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=True)

# # image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_different_angle_.JPG")
# # corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=2.5, area_threshold=5000, approx_tolerance=0.05)