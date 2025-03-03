import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf

model = load_model("./handwritten_digit_cnn.h5")

def custom_crop(image, top=0, bottom=0, left=0, right=0):
    # Ensure we don’t crop beyond the image size
    h, w = image.shape[:2]
    top = min(top, h)
    bottom = min(bottom, h)
    left = min(left, w)
    right = min(right, w)

    # Perform cropping
    return image[top:h-bottom, left:w-right]

def crop_and_preprocess_digit(image,top=0,bottom=0,right=0,left=0,padding =10):
    cropped_img = custom_crop(image,top,bottom,left,right)

    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mean_val = np.mean(gray)
    if mean_val > 128:
        # invert so digit becomes white, background black
        gray = 255 - gray

    # 4) Find bounding box of the white digit
    coords = cv2.findNonZero(gray)  # white is now the digit
    if coords is None:
        # no digit => return a blank 64x64
        return np.zeros((64,64), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    H, W = gray.shape
    print(H,W)
    # Add some padding, but clamp within image bounds
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)
    digit_roi = gray[y1:y2, x1:x2]

    # 5) Letterbox to (64×64) with black padding
    final_64 = letterbox_to_64_black(digit_roi)

    return final_64

def letterbox_to_64_black(gray_img):

    desired = 64
    h, w = gray_img.shape
    print(h,w)

    # scale factor so the digit fits within 64 in both dims
    scale = min(desired / w, desired / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize with a high-quality interpolation
    resized = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new 64×64 black image
    letterboxed = np.zeros((desired, desired), dtype=np.uint8)

    # Compute offset => center the digit
    x_off = (desired - new_w) // 2
    y_off = (desired - new_h) // 2

    # Place the resized digit in center
    letterboxed[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return letterboxed

def final_digit_recognised(image,top,bottom,right,left,valid_digits):
    final_64 = crop_and_preprocess_digit(image,top,bottom,right,left,padding=10)
    print(final_64.shape)

    model_input = final_64.astype(np.float32) / 255.0
    model_input = np.expand_dims(model_input, axis=0)  # shape (1,64,64,3)

    prediction = model.predict(model_input)
    softmax_probs = tf.nn.softmax(prediction).numpy().flatten()
    sub_probs = [softmax_probs[d] for d in valid_digits]
    predicted_digit = valid_digits[np.argmax(sub_probs)]

    return predicted_digit