import cv2
from room_label_detector import detect_room_label_contours_combined

lower_range=(40/2, 5*2.55, 60*2.55)
upper_range=(60/2, 20*2.55, 90*2.55)
image = cv2.imread("/home/kevinbee/Desktop/Indoor_localization_using_room_label_detector/images/office_dark_different_angle.JPG")
corners, number = detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor=4, area_threshold=5000, approx_tolerance=0.05, show_result=False)

print("Corners:")
print(corners)
print("Number:")
print(number)