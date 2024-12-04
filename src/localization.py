import cv2
from room_label_detector import detect_room_label_contours_combined
import json
import os

def localization_with_room_label(corners, numbers, room_labels_library):
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
    [x, y, theta, length, height] = room_labels_library[room_number]



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

print("====================================")
floor_data_path = "/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/data/maps/basic-floor-plan.json"
with open(floor_data_path, 'r') as file:
    floor_data = json.load(file)
    room_labels = {
        name: [room[0], room[1], room[2], room[3], room[4]]
        for name, room in floor_data.get("rooms", {}).items()
    }
print("Available Room Numbers:", room_labels.keys())

print("====================================")
localization_with_room_label(corners, number, room_labels)