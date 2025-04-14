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
# pdf_path = "./Scanned_sheets.pdf"
# pages = convert_from_path(pdf_path, dpi=144)

#Saving the respective pages into directory
os.makedirs("single_sheet", exist_ok=True)

dict_return = {}

# Define how many pixels to crop from top, bottom, left, right:
TOP_CROP = 20
BOTTOM_CROP = 20
LEFT_CROP = 20
RIGHT_CROP = 20

def pdf_processing(pages):
    dict_return = {} # Dictionary with respective SIDs and answers
    for idx, page_img in enumerate(pages):
        # For saving each page in PDF to PNG inside single_sheet directory
        page_path = os.path.join("single_sheet", f"page_{idx}.png")
        page_img.save(page_path, "PNG")

        # Convert from PIL image to NumPy (OpenCV) array
        page_img = np.array(page_img)

        # 2) Crop edges before grayscaling
        h, w = page_img.shape[:2]
        cropped_page_img = page_img[
            TOP_CROP : h - BOTTOM_CROP,
            LEFT_CROP : w - RIGHT_CROP
        ]

        # 3) Convert the cropped portion to grayscale + threshold
        gry = cv2.cvtColor(cropped_page_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gry, 195, 255, cv2.THRESH_BINARY_INV)[1]

        # (Optional) Save the thresholded image for debugging
        gray_path = os.path.join("single_sheet", f"grey_page_{idx}.png")
        cv2.imwrite(gray_path, thresh)

        # 4) Find external contours
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # Store all contour crops for debugging
        sid_boxes_dir = f"sid_boxes_{idx}"
        os.makedirs(sid_boxes_dir, exist_ok=True)

        # Separate bounding boxes into SID boxes and answer option boxes
        sid_boxes = []
        option_boxes = []

        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = w_box * h_box
            # Skip extremely small or extremely large
            if area < 300 or area > 50000:
                continue

            # Classify bounding boxes:
            # For SID boxes: near-square with area between 2200 and 2600.
            if 2200 < area < 2600:
                # Save every SID box for debugging
                # cropped_sid = cropped_page_img[y : y + h_box, x : x + w_box]
                # debug_path = os.path.join(sid_boxes_dir, f"sid_{area}_x{x}_y{y}.png")
                # cv2.imwrite(debug_path, cropped_sid)
                sid_boxes.append((x, y, w_box, h_box))
                print("SID:", x, y, w_box, h_box)
            # For answer option boxes: using the previous area threshold.
            elif 5000 < area < 14000 and w_box < 900:
                option_boxes.append((x, y, w_box, h_box))

        # --- Now handle the SID digits ---
        # Sort SID boxes by left-to-right (primary) then top-to-bottom.
        sid_boxes_sorted = sorted(sid_boxes, key=lambda b: (b[0], b[1]))
        sid_str = ""
        
        # Create a directory for individual SID crops (for debugging)
        sid_dir = f"sid_boxes_{idx}"
        os.makedirs(sid_dir, exist_ok=True)

        for i, (x, y, w_box, h_box) in enumerate(sid_boxes_sorted):
            # Crop each SID digit box
            valid_digits=[0,1,2,3,4,5,6,7,8,9]
            if(i == 3):
                valid_digits=[0,1]
            digit_crop = cropped_page_img[y : y + h_box, x : x + w_box]
            sid_crop_path = os.path.join(sid_dir, f"sid_digit_{i}.png")
            cv2.imwrite(sid_crop_path, digit_crop)

            # Recognize the digit from this crop
            digit = final_digit_recognised(digit_crop,i,sid_dir, 3, 3, 3, 3,valid_digits)
            sid_str += str(digit)

        # If no SID boxes found, assign a fallback key
        if not sid_str:
            sid_str = f"NO_SID_FOUND_PAGE_{idx}"
        
        # Instead of constructing a unique key, check if the SID already exists.
        if sid_str not in dict_return:
            dict_return[sid_str] = []

        # --- Now handle the answer boxes ---
        # Sort answer option boxes by top-to-bottom then left-to-right.
        option_boxes_sorted = sorted(option_boxes, key=lambda b: (b[1], b[0]))
        opt_dir = f"option_boxes_{idx}"
        os.makedirs(opt_dir, exist_ok=True)

        for j, (x, y, w_box, h_box) in enumerate(option_boxes_sorted):
            cropped_opt = cropped_page_img[y : y + h_box, x : x + w_box]
            opt_crop_path = os.path.join(opt_dir, f"option_{j}.png")
            cv2.imwrite(opt_crop_path, cropped_opt)

            # Recognize the digit (assuming valid options are 1-4)
            attempted = final_digit_recognised(cropped_opt,j,opt_dir, 7, 7, 5, 5, valid_digits=[1,2,3,4])
            dict_return[sid_str].append(attempted)
    return dict_return

# Run
# print(pdf_processing(pages))