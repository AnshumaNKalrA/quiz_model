# feature_extractor.py (or add to your training script)
import cv2
import numpy as np
import os
import re

def parse_background_count_from_filename(filename):
    match = re.search(r'_count_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None # Or raise an error/handle appropriately

def extract_features_low_complexity(image_path, background_count_from_filename):
    """
    Extracts a 3-feature vector:
    1. Background Pixel Count (from filename)
    2. Ink Density (from image)
    3. Aspect Ratio of Largest Ink Blob (from image)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    # --- Feature 2: Ink Density ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding - ink becomes white (255), background black (0)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    ink_pixels = np.sum(binary_inv == 255)
    total_pixels = binary_inv.size
    ink_density = ink_pixels / total_pixels if total_pixels > 0 else 0

    # --- Feature 3: Aspect Ratio of Largest Ink Blob ---
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_ratio_largest_blob = 0.0 # Default if no contours or height is 0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 0: # Ensure contour has some area
            x, y, w, h = cv2.boundingRect(largest_contour)
            if h > 0:
                aspect_ratio_largest_blob = w / h
            elif w > 0: # Horizontal line with h=0 or 1
                aspect_ratio_largest_blob = w # or a large predefined number
            # else: it's a point, aspect ratio is ill-defined, 0 is fine
    
    return np.array([background_count_from_filename, ink_density, aspect_ratio_largest_blob])

def load_data_and_extract_features(okay_dir, flagged_dir):
    features_list = []
    labels_list = []

    for dir_path, label in [(okay_dir, 0), (flagged_dir, 1)]: # 0 for okay, 1 for flagged
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory not found {dir_path}")
            continue
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                bg_count = parse_background_count_from_filename(filename)
                if bg_count is None:
                    print(f"Skipping {filename}, could not parse background count.")
                    continue
                
                image_path = os.path.join(dir_path, filename)
                feature_vector = extract_features_low_complexity(image_path, bg_count)
                
                if feature_vector is not None:
                    features_list.append(feature_vector)
                    labels_list.append(label)
    
    return np.array(features_list), np.array(labels_list)