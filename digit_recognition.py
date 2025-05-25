import cv2
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model

# Load the pre-trained model
# Ensure the model file 'handwritten_digit_cnn.h5' is in the same directory as your script
try:
    model = load_model("./handwritten_digit_cnn.h5")
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure 'handwritten_digit_cnn.h5' is in the correct directory.")
    model = None # Set model to None if loading fails

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

def is_struck_out_enhanced(image_norm, option=False, sid_dir=None, index=None):
    """
    Simple detection based on white pixel count range, with different ranges for SID and Option boxes,
    and saves the image for debugging.

    Parameters:
      image_norm: A 2D grayscale image (64x64) normalized to [0,1]
      option: Boolean indicating if the image is from an option box.
      sid_dir: Directory path for saving debug images.
      index: Index of the digit/option being processed for unique filename.

    Returns:
      True if the white pixel count is outside the specified range for the box type; otherwise, False.
    """
    # Convert the normalized image to an 8-bit binary image.
    # Assuming the preprocessing (crop_and_preprocess_digit) already handles inversion
    # such that ink is white (255) and background is black (0).
    img_uint8 = (image_norm * 255).astype(np.uint8)
    # Re-thresholding to ensure clear binary with ink as 255
    # Use THRESH_BINARY_INV + OTSu to ensure ink is white (255) on black (0) background
    _, binary_img = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Count the number of white pixels (non-zero pixels)
    white_pixel_count = cv2.countNonZero(binary_img)

    # --- Save the binary image with pixel count in filename for debugging ---
    # Create a directory for saving debug images if it doesn't exist
    if sid_dir: # Only attempt to save if sid_dir is provided
        debug_output_dir = os.path.join(sid_dir, "debug_pixel_counts")
        os.makedirs(debug_output_dir, exist_ok=True)
        # Save the binary image (which clearly shows ink as white)
        debug_image_path = os.path.join(debug_output_dir, f"digit_{index}_count_{white_pixel_count}.png")
        cv2.imwrite(debug_image_path, binary_img)
        # print(f"Saved debug image for digit {index} with pixel count {white_pixel_count} to: {debug_image_path}") # Uncomment for verbose output
    # --- End Save Logic ---


    # Define the acceptable range for white pixels based on box type
    if option:
        min_white_pixels = 3700
        max_white_pixels = 4000
        box_type = "Option"
    else: # Assuming it's an SID box if not an option
        min_white_pixels = 3400
        max_white_pixels = 4050
        box_type = "SID"


    # Check if the white pixel count is outside the desired range
    if white_pixel_count < min_white_pixels or white_pixel_count > max_white_pixels:
        print(f"{box_type} box: Black pixel count ({white_pixel_count}) is outside the range ({min_white_pixels}-{max_white_pixels}). Flagging.")
        return True
    else:
        # print(f"{box_type} box: White pixel count ({white_pixel_count}) is within the range.")
        return False


def final_digit_recognised(image, i, sid_dir, top, bottom, right, left, valid_digits,option=False):
    """
    Process a digit image:
    - Crop and preprocess it.
    - Use simple detection based on white pixel count (is_struck_out_enhanced), with different ranges for SID and Option.
    - If not marked, use the CNN model to predict the digit.
    """
    # Check if the model was loaded successfully
    if model is None:
        print("Model not loaded, cannot perform digit recognition.")
        return "ERROR" # Or handle as appropriate

    # Preprocess the digit image to a 64x64 grayscale image.
    final_64 = crop_and_preprocess_digit(image, top, bottom, right, left, padding=10)
    if final_64.size == 0 or np.all(final_64 == 0): # Check for empty or all black image after preprocessing
        print(f"Warning: Preprocessed image at index {i} is empty or all black. Skipping recognition.")
        return "N.A"

    # Optionally save for debugging.
    sid_crop_path = os.path.join(sid_dir, f"sid_digit_after_preprocessing_{i}.png")
    cv2.imwrite(sid_crop_path, final_64)
    print("Preprocessed digit shape:", final_64.shape)

    # Normalize image to [0,1].
    norm_img = final_64.astype(np.float32) / 255.0

    # Use the simple detection based on white pixel count range.
    # Pass the 'option' flag to is_struck_out_enhanced to use the correct range.
    # Pass sid_dir and index for potential debug saving within is_struck_out_enhanced.
    if is_struck_out_enhanced(norm_img, option=option, sid_dir=sid_dir, index=i):
        # print(f"Digit at index {i} appears to be marked (white pixel count check).")
        return "N.A"

    # --- CNN Prediction (if not marked by the simple check) ---
    # This part is reached only if is_struck_out_enhanced did NOT flag the image.

    # Prepare the image for the model: add channel and batch dimensions.
    model_input = np.expand_dims(norm_img, axis=-1)
    model_input = np.expand_dims(model_input, axis=0)

    # Perform CNN prediction
    prediction = model.predict(model_input, verbose=0) # Set verbose to 0 to avoid printing progress bar
    softmax_probs = tf.nn.softmax(prediction).numpy().flatten()

    # Find the predicted digit among the valid ones
    sub_probs = [softmax_probs[d] for d in valid_digits]
    predicted_digit = valid_digits[np.argmax(sub_probs)]

    # Original option-specific check after prediction
    # This check is kept as in the original file structure.
    if(option):
        # Check if the overall most probable digit is the same as the predicted digit among valid ones
        # This helps catch cases where a non-valid digit might have a higher probability overall.
        if(np.argmax(softmax_probs) != predicted_digit):

            return "N.A"

    # If not flagged by the simple check or the option check (if applicable), return the predicted digit
    return predicted_digit