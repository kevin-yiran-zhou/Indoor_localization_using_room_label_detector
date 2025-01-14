import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import time


def reorder_corners(corners):
    top_right, top_left, bottom_left, bottom_right = corners
    points = np.array([top_right, top_left, bottom_left, bottom_right])
    y_sorted = points[np.argsort(points[:, 1])]
    top_points = y_sorted[:2]
    bottom_points = y_sorted[2:]
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    return np.array([top_right, top_left, bottom_left, bottom_right])


def detect_number(image, corners, show_result):
    reordered_corners = reorder_corners(corners)
    top_right, top_left, bottom_left, bottom_right = reordered_corners

    width = max(np.linalg.norm(np.array(top_right) - np.array(top_left)),
                np.linalg.norm(np.array(bottom_right) - np.array(bottom_left)))
    height = max(np.linalg.norm(np.array(top_left) - np.array(bottom_left)),
                 np.linalg.norm(np.array(top_right) - np.array(bottom_right)))
    
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (int(width), int(height)))
    if show_result:
        cv2.imshow('Warped Image', warped)
        # while cv2.getWindowProperty('Warped Image', cv2.WND_PROP_VISIBLE) >= 1:
        #     key = cv2.waitKey(1)
        #     if key == 27:
        #         break

    # OCR
    image = Image.fromarray(warped)
    gray_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.8)
    threshold_image = enhanced_image.point(lambda p: p > 75 and 255)
    sharpened_image = threshold_image.filter(ImageFilter.SHARPEN)
    if show_result:
        cv2.imshow('Sharpened Image', np.array(sharpened_image))
        # while cv2.getWindowProperty('Sharpened Image', cv2.WND_PROP_VISIBLE) >= 1:
        #     key = cv2.waitKey(1)
        #     if key == 27:
        #         break

    ## Detect only digits
    # text = pytesseract.image_to_string(sharpened_image, config='--oem 3 --psm 6 outputbase digits')
    # Detect digits and letters
    text = pytesseract.image_to_string(sharpened_image, config='--oem 3 --psm 6')

    text = text.replace(" ", "").replace("\n", "")
    return text, reordered_corners


def detect_room_label_contours(image, resize_factor, area_threshold, approx_tolerance, show_result):
    start_time = time.time()
    # Resize the image
    image = cv2.resize(image, (int(image.shape[1] / resize_factor), int(image.shape[0] / resize_factor)))
    
    # Apply bilateral filter for noise reduction while keeping edges sharp
    filtered_image = cv2.bilateralFilter(image, 15, 80, 80)
    
    # Convert to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    
    # Apply dilation followed by erosion to enhance the shapes
    kernel = np.ones((5, 5), np.float32) / 49
    dilated = cv2.dilate(gray, kernel, iterations=3)
    
    # Apply erosion to remove noise
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Detect edges using Canny Edge Detector
    thr1, thr2 = 50, 200
    edged = cv2.Canny(eroded, thr1, thr2)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangle_corners = []
    result_corners = []
    result_texts = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > area_threshold:
            approx = cv2.approxPolyDP(cnt, approx_tolerance * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                corners = [point[0] for point in approx]
                rectangle_corners.append(corners)

                OCR_result, reordered_corners = detect_number(image, corners, show_result)

                if sum(char.isdigit() for char in OCR_result) >= 1:
                # if len(OCR_result) >= 4:
                    if show_result:
                        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                        cv2.putText(image, OCR_result, tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    result_corners.append(reordered_corners)
                    result_texts.append(OCR_result)
                else:
                    if show_result:
                        cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)
                        cv2.putText(image, "No numbers", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                if show_result:
                    cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)
                    cv2.putText(image, "Not rectangle", tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # pass

    print("Time taken:", time.time() - start_time)
    if show_result:
        cv2.imshow('Detected Room Labels (Contours)', image)
        while cv2.getWindowProperty('Detected Room Labels (Contours)', cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(1)
            if key == 27:
                break
        cv2.destroyAllWindows()
    return result_corners, result_texts


def detect_room_label_contours_hsv(image, lower_range, upper_range, resize_factor, area_threshold, approx_tolerance, show_result):
    start_time = time.time()

    # Resize the image
    image = cv2.resize(image, (int(image.shape[1] / resize_factor), int(image.shape[0] / resize_factor)))
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the HSV range
    mask = cv2.inRange(hsv_image, lower_range, upper_range)

    if show_result:
        # Show the mask (binary image)
        cv2.imshow("Mask", mask)
        # Wait for user interaction to close the windows
        while cv2.getWindowProperty('Mask', cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(1)
            if key == 27:
                break
        cv2.destroyAllWindows()
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangle_corners = []
    result_corners = []
    result_texts = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > area_threshold:
            approx = cv2.approxPolyDP(cnt, approx_tolerance * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                corners = [point[0] for point in approx]
                rectangle_corners.append(corners)

                OCR_result, reordered_corners = detect_number(image, corners, show_result)

                if sum(char.isdigit() for char in OCR_result) >= 0:
                # if len(OCR_result) >= 4:
                    if show_result:
                        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                        cv2.putText(image, OCR_result, tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    result_corners.append(reordered_corners)
                    result_texts.append(OCR_result)
                else:
                    if show_result:
                        cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)
                        cv2.putText(image, "No numbers", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                if show_result:
                    cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)
                    cv2.putText(image, "Not rectangle", tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # pass

    print("Time taken:", time.time() - start_time)
    
    # Display the processed image
    if show_result:
        cv2.imshow("Detected Room Labels (HSV)", image)
        while cv2.getWindowProperty('Detected Room Labels (HSV)', cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(1)
            if key == 27:
                break
        cv2.destroyAllWindows()
    return result_corners, result_texts


def detect_room_label_contours_combined(image, lower_range, upper_range, resize_factor, area_threshold, approx_tolerance, show_result):
    corners, number = detect_room_label_contours(image, resize_factor, area_threshold, approx_tolerance, show_result)
    if len(number) == 0:
        print("No numbers detected using detect_room_label_contours. Trying detect_room_label_contours_hsv.")
        corners, number = detect_room_label_contours_hsv(image, lower_range, upper_range, resize_factor, area_threshold, approx_tolerance, show_result)
        if len(number) == 0:
            print("No numbers detected using both detect_room_label_contours either. Returning empty result.")
        else:
            print("Numbers detected using detect_room_label_contours_hsv.")
    else:
        print("Numbers detected using detect_room_label_contours.")
    print("Number:", number)
    print("====================================")
    return corners, number