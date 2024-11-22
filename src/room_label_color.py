import cv2
import numpy as np

def detect_grey_labels_hsv(image, lower_range, upper_range, resize_factor=5):
    # Resize the image
    resized_image = cv2.resize(image, (int(image.shape[1] / resize_factor), int(image.shape[0] / resize_factor)))
    
    # Convert the resized image to HSV color space
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    
    # Apply a mask to isolate grey regions
    mask = cv2.inRange(hsv_image, lower_range, upper_range)
    
    # Display the masked image
    cv2.imshow("Grey Mask (HSV)", mask)
    while cv2.getWindowProperty('Grey Mask (HSV)', cv2.WND_PROP_VISIBLE) >= 1:
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()


lower_range=(40/2, 5*2.55, 60*2.55)
upper_range=(60/2, 20*2.55, 90*2.55)


# Read the input image
image = cv2.imread("/home/kevinbee/Desktop/room_label_detector/images/office.JPG")
detect_grey_labels_hsv(image, lower_range, upper_range, resize_factor=5)

image = cv2.imread("/home/kevinbee/Desktop/room_label_detector/images/office_dark.JPG")
detect_grey_labels_hsv(image, lower_range, upper_range, resize_factor=5)

image = cv2.imread("/home/kevinbee/Desktop/room_label_detector/images/office_dark_different_angle.JPG")
detect_grey_labels_hsv(image, lower_range, upper_range, resize_factor=5)

image = cv2.imread("/home/kevinbee/Desktop/room_label_detector/images/office_rotated.JPG")
detect_grey_labels_hsv(image, lower_range, upper_range, resize_factor=5)

image = cv2.imread("/home/kevinbee/Desktop/room_label_detector/images/office_different_angle.JPG")
detect_grey_labels_hsv(image, lower_range, upper_range, resize_factor=5)