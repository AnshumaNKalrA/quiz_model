import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
from keras.models import load_model
from pdf2image import convert_from_path
from PIL import Image
import imutils
from digit_recognition import final_digit_recognised
import json

# Loading the digit recognition model trained on kaggle dataset
model = load_model("./handwritten_digit_cnn.h5")

# 1) Convert each page of the PDF into a PIL image
pdf_path = "./Scanned_sheets.pdf"
pages = convert_from_path(pdf_path, dpi=144)

os.makedirs("single_sheet", exist_ok=True)

dict_return = {}

# Define how many pixels to crop from top, bottom, left, right:
TOP_CROP = 20
BOTTOM_CROP = 20
LEFT_CROP = 20
RIGHT_CROP = 20

def pdf_processing(pages):
    for idx, page_img in enumerate(pages):
        # For saving each of the pages in PDF to PNG
        page_path = os.path.join("single_sheet", f"page_{idx}.png")
        page_img.save(page_path, "PNG")

        # Convert from PIL image to NumPy (OpenCV) array
        page_img = np.array(page_img)

        # 2) Crop edges before grayscaling
        h, w = page_img.shape[:2]
        # Ensure we don't go out of bounds
        cropped_page_img = page_img[
            TOP_CROP : h - BOTTOM_CROP,
            LEFT_CROP : w - RIGHT_CROP
        ]

        # 3) Convert the cropped portion to grayscale
        gry = cv2.cvtColor(cropped_page_img, cv2.COLOR_BGR2GRAY)

        # 4) Threshold + contour detection
        thresh = cv2.threshold(gry, 195, 255, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # Save the thresholded image (optional)
        gray_path = os.path.join("single_sheet", f"grey_page_{idx}.png")
        cv2.imwrite(gray_path, thresh)

        # 5) Find target bounding boxes & crop
        mask = np.ones(cropped_page_img.shape[:2], dtype="uint8") * 255
        boxes = []

        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = w_box * h_box

            # Create output directory for the cropped rectangles of this page
            output_dir = f"cropped_rects_{idx}"
            os.makedirs(output_dir, exist_ok=True)

            # Same bounding-box filter (adjust if needed)
            if area > 5000 and area < 14000 and w_box < 900:
                cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), (0, 0, 255), -1)
                boxes.append((x, y, w_box, h_box, area))
                print(area,w_box,h_box)

        # 6) Sort the boxes and save them
        boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))

        for index, (x, y, w_box, h_box, area) in enumerate(boxes_sorted):
            # Fill rectangle on mask
            cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), (0, 0, 255), -1)
                
            # Crop the detected box from the *cropped* page image
            cropped = cropped_page_img[y : y + h_box, x : x + w_box]

            # Save each detected rectangle
            cropped_path = os.path.join(output_dir, f"cropped_{index}.png")
            cv2.imwrite(cropped_path, cropped)
            print(f"Saved cropped rectangle (y={y}, x={x}) to: {cropped_path}")

        # 7) Optional: bitwise-and to visualize the final mask on the cropped page
        # res_final = cv2.bitwise_and(cropped_page_img, cropped_page_img, 
        #                             mask=cv2.bitwise_not(mask))

        #Recognising SID
        SID_image_path = f'cropped_rects_{idx}/cropped_0.png'
        SID_image = cv2.imread(SID_image_path)
        if SID_image is None:
            raise FileNotFoundError(f"Unable to load {SID_image_path}.")
        
        gry = cv2.cvtColor(SID_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gry, 195, 255, cv2.THRESH_BINARY_INV)[1]

        # Suppose 'thr' is your padded, thresholded image showing white lines on black background.
        # 1) Use morphological operations to ensure vertical & horizontal lines become closed loops:

        # Close vertical gaps with a vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        closed_vert = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel)

        # Close horizontal gaps with a horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        closed_both = cv2.morphologyEx(closed_vert, cv2.MORPH_CLOSE, horizontal_kernel)

        os.makedirs(f"sid_digits_{idx}", exist_ok=True)
        cropped_path = os.path.join(f"sid_digits_{idx}", f"sid_{idx}.png")
        cv2.imwrite(cropped_path, closed_both)

        # 2) Find contours **with RETR_TREE** so we see inner contours:
        cnts = cv2.findContours(closed_both.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # If SID_image is the original image to crop from (NOT thresholded):
        sid_boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            # Filter out very small noise or extremely large bounding boxes
            if area < 300 or area > 50000:
                continue
            sid_boxes.append((x, y, w, h))

        # Sort the boxes left-right
        sid_boxes_sorted = sorted(sid_boxes, key=lambda b: b[0])

        # 3) Save each cropped rectangle
        output_dir = "cropped_rect"
        os.makedirs(output_dir, exist_ok=True)

        SID_list = list()
        sid = ""
        for i, (x, y, w, h) in enumerate(sid_boxes_sorted):
            if(i==0):
                continue
            if(i==9):
                break
            cropped = SID_image[y:y+h, x:x+w]  # crop from original
            save_path = f"cropped_rect/cropped_{idx}_{i}.png"
            cv2.imwrite(save_path,cropped)
            digit = final_digit_recognised(cropped,2,2,2,2,valid_digits=[1,2,3,4,5,6,7,8,9,0])
            sid += str(digit)
        dict_return[sid] = list()

        for index, (x, y, w_box, h_box, area) in enumerate(boxes_sorted):
            if(area > 8000):
                continue
            # Fill rectangle on mask
            # cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), (0, 0, 255), -1)
                
            # Crop the detected box from the *cropped* page image
            cropped = cropped_page_img[y : y + h_box, x : x + w_box]
            attempted = final_digit_recognised(cropped,7,7,5,5,valid_digits=[1,2,3,4])
            dict_return[sid].append(attempted)
    
    json_return = json.dumps(dict_return)
    return json_return
