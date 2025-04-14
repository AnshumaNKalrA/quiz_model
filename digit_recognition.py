import cv2
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model

model = load_model("./handwritten_digit_cnn.h5")

def custom_crop(image, top=0, bottom=0, left=0, right=0):
    h, w = image.shape[:2]
    top = min(top, h)
    bottom = min(bottom, h)
    left = min(left, w)
    right = min(right, w)
    return image[top:h-bottom, left:w-right]

def letterbox_to_64_black(gray_img):
    desired = 64
    h, w = gray_img.shape
    scale = min(desired / w, desired / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    letterboxed = np.zeros((desired, desired), dtype=np.uint8)
    x_off = (desired - new_w) // 2
    y_off = (desired - new_h) // 2
    letterboxed[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return letterboxed

def crop_and_preprocess_digit(image, top=0, bottom=0, right=0, left=0, padding=10):
    cropped_img = custom_crop(image, top, bottom, left, right)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # Use Otsu thresholding after binary inversion
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Invert if necessary, based on mean intensity
    if np.mean(gray) > 128:
        gray = 255 - gray
    coords = cv2.findNonZero(gray)
    if coords is None:
        return np.zeros((64, 64), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    H, W = gray.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)
    digit_roi = gray[y1:y2, x1:x2]
    final_64 = letterbox_to_64_black(digit_roi)
    return final_64

def is_struck_out_enhanced(image_norm):
    """
    Enhanced strike-through detection that handles wavy/haphazard strokes.
    
    Parameters:
      image_norm: A 2D grayscale image (64x64) normalized to [0,1]
      
    Returns:
      True if evidence of a strike-out stroke is found; otherwise, False.
    """
    # Convert the normalized image to an 8-bit binary image.
    img_uint8 = (image_norm * 255).astype(np.uint8)
    _, binary_img = cv2.threshold(img_uint8, 128, 255, cv2.THRESH_BINARY)
    
    # --- Method 1: Hough Transform Based Detection ---
    # Use Canny edge detection to get edges
    edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)
    # HoughLinesP to detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                            minLineLength=int(0.6 * binary_img.shape[1]), 
                            maxLineGap=5)
    if lines is not None:
        width = binary_img.shape[1]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if line_length < 0.6 * width:
                continue
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            # Check if line is near horizontal or near a 45° diagonal (adjust tolerances as needed)
            if (abs(angle) < 30) or (abs(abs(angle) - 45) < 20):
                return True

    # --- Method 2: Contour Analysis within a Central Band ---
    # Define a central band in the vertical direction.
    h, w = binary_img.shape
    band_height = h // 3  # use one-third of the height
    band = binary_img[(h // 2 - band_height // 2):(h // 2 + band_height // 2), :]
    # Find contours in the central band.
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Heuristic: if the contour spans most of the width but is relatively thin, it may be a strike.
        if cw > 0.6 * w and ch < band_height * 0.6:
            # Optionally, further measures (e.g. aspect ratio or solidity) can be added.
            return True

    return False

def final_digit_recognised(image, i, sid_dir, top, bottom, right, left, valid_digits,option=False):
    """
    Process a digit image:
    - Crop and preprocess it.
    - Use advanced strike-out detection to determine if it was cancelled.
    - If not cancelled, use the CNN model to predict the digit.
    """
    # Preprocess the digit image to a 64x64 grayscale image.
    final_64 = crop_and_preprocess_digit(image, top, bottom, right, left, padding=10)
    # Optionally save for debugging.
    sid_crop_path = os.path.join(sid_dir, f"sid_digit_after_preprocessing_{i}.png")
    cv2.imwrite(sid_crop_path, final_64)
    print("Preprocessed digit shape:", final_64.shape)
    
    # Normalize image to [0,1].
    norm_img = final_64.astype(np.float32) / 255.0
    
    # Use the advanced strike-out detector.
    if is_struck_out_enhanced(norm_img):
        print(f"Digit at index {i} appears to be struck-out.")
        return "N.A"
    
    # Prepare the image for the model: add channel and batch dimensions.
    model_input = np.expand_dims(norm_img, axis=-1)
    model_input = np.expand_dims(model_input, axis=0)
    
    prediction = model.predict(model_input)
    softmax_probs = tf.nn.softmax(prediction).numpy().flatten()
    sub_probs = [softmax_probs[d] for d in valid_digits]
    predicted_digit = valid_digits[np.argmax(sub_probs)]
    if(option):
        if(np.argmax(softmax_probs) != predicted_digit):
            return "N.A"
    return predicted_digit
