import os
import cv2
import numpy as np
from keras.models import load_model
from pdf2image import convert_from_path
# Assuming final_digit_recognised is needed later, but commented out for now
# from digit_recognition import final_digit_recognised

# Load the model (assuming it's in the same directory), commented out for now
# try:
#     model = load_model("./handwritten_digit_cnn.h5")
# except Exception as e:
#     print(f"Error loading the model: {e}")
#     print("Please ensure 'handwritten_digit_cnn.h5' is in the correct directory.")
#     model = None # Set model to None if loading fails


pdf_path = "./scanned.pdf"
# pdf_path = "./Scanned_sheets.pdf" # Uncomment and update if the file name is different

# Check if the PDF file exists
if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
else:
    try:
        pages = convert_from_path(pdf_path, dpi=144)
        os.makedirs("single_sheet", exist_ok=True)
        TOP_CROP = 20
        BOTTOM_CROP = 20
        LEFT_CROP = 20
        RIGHT_CROP = 20

        def pdf_processing(pages):
            dict_return = {}
            for idx, page_img in enumerate(pages):
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
                # Using a fixed threshold as in the original code
                thresh = cv2.threshold(gry, 195, 255, cv2.THRESH_BINARY_INV)[1]

                gray_path = os.path.join("single_sheet", f"grey_page_{idx}.png")
                cv2.imwrite(gray_path, thresh)

                # Find contours, using thresh.copy() as findContours can modify the input image
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                sid_boxes = []
                option_boxes = []

                # Create a directory to save contours for the current page
                contour_output_dir = os.path.join("single_sheet", f"page_{idx}_contours")
                os.makedirs(contour_output_dir, exist_ok=True)

                for c_idx, c in enumerate(contours):
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    area = w_box * h_box
                    aspect_ratio = w_box / h_box if h_box > 0 else 0

                    # Save each contour image with its properties in the filename
                    # Ensure cropping coordinates are within bounds
                    y_end = min(y + h_box, cropped_page_img.shape[0])
                    x_end = min(x + w_box, cropped_page_img.shape[1])
                    contour_img = cropped_page_img[y : y_end, x : x_end]

                    # Check if the cropped contour image is not empty before saving
                    if contour_img.size > 0:
                         contour_filename = os.path.join(contour_output_dir, f"contour_{c_idx}_x{x}_y{y}_w{w_box}_h{h_box}_area{area}_ar{aspect_ratio:.2f}.png")
                         cv2.imwrite(contour_filename, contour_img)
                    else:
                         print(f"Warning: Empty contour image for contour {c_idx} on page {idx}. Skipping save.")


                    # Original filtering logic (keep this to continue with the rest of the process)
                    # You will likely need to adjust these thresholds based on the saved contour properties
                    if area < 300 or area > 50000:
                        continue

                    if 2200 < area < 2850:
                        sid_boxes.append((x, y, w_box, h_box))
                    elif 5000 < area < 14000 and w_box < 900:
                        option_boxes.append((x, y, w_box, h_box))

                # The rest of the pdf_processing function remains the same
                # Drawing bounding boxes on the annotated page
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

                # Sorting SID boxes and preparing for digit recognition (currently commented out)
                sid_boxes_sorted = sorted(sid_boxes, key=lambda b: (b[0], b[1]))
                sid_str = ""
                sid_dir = f"sid_boxes_{idx}"
                os.makedirs(sid_dir, exist_ok=True)

                # The loops for digit and option recognition are commented out as requested previously.
                # If you uncomment them, ensure the 'model' is loaded and final_digit_recognised is imported
                # from digit_recognition.py

                # for i, (x, y, w_box, h_box) in enumerate(sid_boxes_sorted):
                #    valid_digits = [0,1,2,3,4,5,6,7,8,9]
                #    if(i == 3):
                #        valid_digits = [0,1]
                #    y_end = min(y + h_box, cropped_page_img.shape[0])
                #    x_end = min(x + w_box, cropped_page_img.shape[1])
                #    digit_crop = cropped_page_img[y : y_end, x : x_end]

                #    if digit_crop.size == 0:
                #        print(f"Warning: Empty SID crop at index {i} for page {idx}. Skipping.")
                #        sid_str += "?"
                #        continue

                #    sid_crop_path = os.path.join(sid_dir, f"sid_digit_{i}.png")
                #    cv2.imwrite(sid_crop_path, digit_crop)

                #    try:
                #        # Ensure final_digit_recognised function and its dependencies are available
                #        digit = final_digit_recognised(digit_crop, i, sid_dir, 3, 3, 3, 3, valid_digits)
                #        sid_str += str(digit)
                #    except Exception as e:
                #        print(f"Error recognizing SID digit {i} on page {idx}: {e}")
                #        sid_str += "?"

                # if not sid_str:
                #     sid_str = f"NO_SID_FOUND_PAGE_{idx}"

                # if sid_str not in dict_return:
                #     dict_return[sid_str] = []

                # option_boxes_sorted = sorted(option_boxes, key=lambda b: (b[1], b[0]))
                # opt_dir = f"option_boxes_{idx}"
                # os.makedirs(opt_dir, exist_ok=True)

                # for j, (x, y, w_box, h_box) in enumerate(option_boxes_sorted):
                #    y_end = min(y + h_box, cropped_page_img.shape[0])
                #    x_end = min(x + w_box, cropped_page_img.shape[1])
                #    cropped_opt = cropped_page_img[y : y_end, x : x_end]

                #    if cropped_opt.size == 0:
                #        print(f"Warning: Empty Option crop at index {j} for page {idx}. Skipping.")
                #        dict_return[sid_str].append("?")
                #        continue

                #    opt_crop_path = os.path.join(opt_dir, f"option_{j}.png")
                #    cv2.imwrite(opt_crop_path, cropped_opt)

                #    try:
                #        # Ensure final_digit_recognised function and its dependencies are available
                #        attempted = final_digit_recognised(cropped_opt,j,opt_dir, 7, 7, 5, 5, valid_digits=[1,2,3,4],option=True)
                #        dict_return[sid_str].append(attempted)
                #    except Exception as e:
                #        print(f"Error recognizing Option {j} on page {idx}: {e}")
                #        dict_return[sid_str].append("?")

            # Returning an empty dictionary as the recognition part is commented out
            # return dict_return
            return {}


        results = pdf_processing(pages)
        print(results)

    except Exception as e:
        print(f"An unexpected error occurred during PDF processing: {e}")