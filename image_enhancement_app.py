import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set Streamlit layout to wide for better image display
st.set_page_config(layout="wide")

# Adjust image brightness by adding a constant value (beta)
def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)

# Perform contrast stretching to improve image contrast
def contrast_stretching(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    min_val, max_val = np.min(gray), np.max(gray)
    stretched = ((gray - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)

# Apply global histogram equalization to enhance contrast
def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)


# Set Streamlit title and image uploader
st.title("Image Enhancement with OpenCV")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
