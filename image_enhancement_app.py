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

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_histogram_equalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

# Apply Gaussian Blur to reduce image noise
def gaussian_denoising(image, kernel_size=(5,5), sigma=1.5):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Apply Median Blur to remove salt-and-pepper noise
def median_denoising(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


# Set Streamlit title and image uploader
st.title("Image Enhancement with OpenCV")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Resize uploaded image to fit within 600 pixels if it's too large
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))

    max_dim = 600
    if max(image_np.shape[:2]) > max_dim:
        scale = max_dim / max(image_np.shape[:2])
        image_np = cv2.resize(image_np, (int(image_np.shape[1]*scale), int(image_np.shape[0]*scale)))
        

# Create dropdown menu to select image enhancement technique
enhancement_type = st.selectbox("Select enhancement type", [
    'Adjust Brightness',
    'Contrast Stretching',
    'Histogram Equalization',
    'CLAHE Equalization',
    'Gaussian Denoising',
    'Median Denoising',
    'Sharpening',
    'Color Correction',
    'White Balance'
])

    if enhancement_type == 'Adjust Brightness':
        beta = st.slider("Brightness Level", -100, 100, 30)
        enhanced_image = adjust_brightness(image_np, beta)

    elif enhancement_type == 'Contrast Stretching':
        enhanced_image = contrast_stretching(image_np)

    elif enhancement_type == 'Histogram Equalization':
        enhanced_image = histogram_equalization(image_np)

    elif enhancement_type == 'CLAHE Equalization':
    enhanced_image = clahe_histogram_equalization(image_np)

