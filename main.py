import os
import cv2
import numpy as np
from keras.models import load_model
from pdf2image import convert_from_path
from digit_recognition import final_digit_recognised # Assuming this file exists and is correct

# Load model and define paths
model = load_model("./handwritten_digit_cnn.h5")
pdf_path = "./scanned.pdf" # This will be used in the try-except block later

# Create directories
os.makedirs("single_sheet", exist_ok=True)
CONTOUR_IMAGES_DIR = "contour_images_with_areas"  # Define the new directory name
os.makedirs(CONTOUR_IMAGES_DIR, exist_ok=True)    # Create the new directory

# Define cropping constants
TOP_CROP = 20
BOTTOM_CROP = 20
LEFT_CROP = 20
RIGHT_CROP = 20

def pdf_processing(pages_list): # Renamed 'pages' to 'pages_list' for clarity
    dict_return = {}
    for idx, page_img in enumerate(pages_list):
        page_path = os.path.join("single_sheet", f"page_{idx}.png")
        page_img.save(page_path, "PNG")

        page_img_np = np.array(page_img)

        h, w = page_img_np.shape[:2]
        if h <= (TOP_CROP + BOTTOM_CROP) or w <= (LEFT_CROP + RIGHT_CROP):
            print(f"Warning: Page {idx} is too small for cropping. Skipping.")
            continue

        cropped_page_img = page_img_np[
            TOP_CROP : h - BOTTOM_CROP,
            LEFT_CROP : w - RIGHT_CROP
        ]

        gry = cv2.cvtColor(cropped_page_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gry, 195, 255, cv2.THRESH_BINARY_INV)[1]

        gray_path = os.path.join("single_sheet", f"grey_page_{idx}.png")
        cv2.imwrite(gray_path, thresh)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        sid_boxes = []
        option_boxes = []

        # Iterate through contours with an index for unique naming
        for c_idx, c in enumerate(contours):
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = w_box * h_box  # Area of the bounding box

            # Filter contours based on area (same as original code)
            if area < 300 or area > 50000:
                continue

            # --- New: Save the contour image (cropped bounding box) ---
            # Crop the region of the contour from the cropped page image
            contour_image_crop = cropped_page_img[y : y + h_box, x : x + w_box]

            if contour_image_crop.size > 0:
                # Construct filename: page_idx_contour_c_idx_area_value.png
                contour_filename = f"page_{idx}_contour_{c_idx}_area_{int(area)}.png"
                contour_save_path = os.path.join(CONTOUR_IMAGES_DIR, contour_filename)
                cv2.imwrite(contour_save_path, contour_image_crop)
                # print(f"Saved contour image to: {contour_save_path}") # Optional: uncomment for verbose logging
            else:
                print(f"Warning: Page {idx}, Contour {c_idx} - Empty crop, not saving contour image.")
            # --- End new code for saving contour image ---

            # Original logic for classifying contours into SID or option boxes
            if 2100 < area < 2900:
                sid_boxes.append((x, y, w_box, h_box))
            elif 5000 < area < 14000 and w_box < 900:
                option_boxes.append((x, y, w_box, h_box))

        annotated_page = cropped_page_img.copy()
        box_color = (0, 255, 0)
        box_thickness = 2

        for (x, y, w_box, h_box) in sid_boxes:
            pt1 = (x, y)
            pt2 = (x + w_box, y + h_box)
            cv2.rectangle(annotated_page, pt1, pt2, box_color, box_thickness)

        for (x, y, w_box, h_box) in option_boxes:
            pt1 = (x, y)
            pt2 = (x + w_box, y + h_box)
            cv2.rectangle(annotated_page, pt1, pt2, box_color, box_thickness)

        annotated_save_path = os.path.join("single_sheet", f"annotated_page_{idx}.png")
        cv2.imwrite(annotated_save_path, annotated_page)
        print(f"Saved annotated image for page {idx} to: {annotated_save_path}")

        sid_boxes_sorted = sorted(sid_boxes, key=lambda b: (b[0], b[1]))
        sid_str = ""
        sid_dir = f"sid_boxes_{idx}"
        os.makedirs(sid_dir, exist_ok=True)

        for i, (x, y, w_box, h_box) in enumerate(sid_boxes_sorted):
            valid_digits = [0,1,2,3,4,5,6,7,8,9]
            if(i == 3): # Assuming this condition is specific to your use case
                valid_digits = [0,1]
            y_end = min(y + h_box, cropped_page_img.shape[0])
            x_end = min(x + w_box, cropped_page_img.shape[1])
            digit_crop = cropped_page_img[y : y_end, x : x_end]

            if digit_crop.size == 0:
                print(f"Warning: Empty SID crop at index {i} for page {idx}. Skipping.")
                sid_str += "?"
                continue

            sid_crop_path = os.path.join(sid_dir, f"sid_digit_{i}.png")
            cv2.imwrite(sid_crop_path, digit_crop)

            try:
                digit = final_digit_recognised(digit_crop, i, sid_dir, 3, 3, 3, 3, valid_digits)
                sid_str += str(digit)
            except Exception as e:
                print(f"Error recognizing SID digit {i} on page {idx}: {e}")
                sid_str += "?"

        if not sid_str:
            sid_str = f"NO_SID_FOUND_PAGE_{idx}"

        if sid_str not in dict_return:
            dict_return[sid_str] = []

        option_boxes_sorted = sorted(option_boxes, key=lambda b: (b[1], b[0]))
        opt_dir = f"option_boxes_{idx}"
        os.makedirs(opt_dir, exist_ok=True)

        for j, (x, y, w_box, h_box) in enumerate(option_boxes_sorted):
            y_end = min(y + h_box, cropped_page_img.shape[0])
            x_end = min(x + w_box, cropped_page_img.shape[1])
            cropped_opt = cropped_page_img[y : y_end, x : x_end]

            if cropped_opt.size == 0:
                print(f"Warning: Empty Option crop at index {j} for page {idx}. Skipping.")
                dict_return[sid_str].append("?")
                continue

            opt_crop_path = os.path.join(opt_dir, f"option_{j}.png")
            cv2.imwrite(opt_crop_path, cropped_opt)

            try:
                attempted = final_digit_recognised(cropped_opt,j,opt_dir, 7, 7, 5, 5, valid_digits=[1,2,3,4],option=True)
                dict_return[sid_str].append(attempted)
            except Exception as e:
                print(f"Error recognizing Option {j} on page {idx}: {e}")
                dict_return[sid_str].append("?")

    return dict_return

# Main execution block
try:
    # pdf_path variable is already defined globally
    # TOP_CROP, BOTTOM_CROP, etc. are also defined globally
    # os.makedirs("single_sheet", exist_ok=True) # Already done globally

    # Convert PDF to images
    pages_from_pdf = convert_from_path(pdf_path, dpi=144) # Use the global pdf_path

    # Process the PDF pages
    results = pdf_processing(pages_from_pdf)
    print("Processing Results:", results)

except FileNotFoundError:
    print(f"Error: PDF file not found at {pdf_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")