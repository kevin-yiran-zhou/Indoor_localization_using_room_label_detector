import cv2
from room_label_detector import detect_room_label_contours_combined
import json
import os

# def 





hsv_file_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/data/hsv_colors.json"
with open(hsv_file_path, 'r') as file:
    hsv_colors = json.load(file)
lower_range = tuple(hsv_colors["gray"]["lower"])
upper_range = tuple(hsv_colors["gray"]["upper"])
image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_dark_different_angle.JPG")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=False)

print("Corners:")
print(corners)
print("Number:")
print(number)