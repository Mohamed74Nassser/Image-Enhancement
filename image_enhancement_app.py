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
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) ## L=>Lightness, A=>(green-red), B=>(blue-yellow)	الفرق بين اللونين
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
    
# Sharpen the image using a custom kernel to enhance edges
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Enhance image colors by adjusting the saturation in HSV color space
def color_correction(image, saturation_factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255) 
    ## H = Hue (لون) S = Saturation (تشبع اللون) V = Value (الإضاءة)
    ## hsv[:, :, 1] => row = الصف col = العمود channel = القناة (0 أو 1 أو 2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Adjust white balance based on LAB color statistics to correct color tones
def white_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])
    lab[:, :, 1] -= ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] -= ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

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

    elif enhancement_type == 'Gaussian Denoising':
        kernel_size = st.slider("Kernel Size (odd)", 1, 21, 5, step=2)
        sigma = st.slider("Sigma", 0.1, 5.0, 1.5)
        enhanced_image = gaussian_denoising(image_np, (kernel_size, kernel_size), sigma)

    elif enhancement_type == 'Median Denoising':
        kernel_size = st.slider("Kernel Size (odd)", 1, 21, 5, step=2)
        enhanced_image = median_denoising(image_np, kernel_size)

    elif enhancement_type == 'Sharpening':
        enhanced_image = sharpen_image(image_np)

    elif enhancement_type == 'Color Correction':
        saturation_factor = st.slider("Saturation Level", 0.0, 3.0, 1.5)
        enhanced_image = color_correction(image_np, saturation_factor)

    elif enhancement_type == 'White Balance':
        enhanced_image = white_balance(image_np)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original Image ")
        st.image(image, width=500)
    with col2:
        st.markdown(f"### Enhanced: {enhancement_type} ")
        st.image(enhanced_image, width=500)

else:
    st.warning("No image uploaded yet. Please upload an image to display it.")

# Display credits at the bottom
st.markdown(""" 
    <h4 style='text-align: center;'>
        <br>
        <br>
        Developed By 
        <br>
        طارق ممدوح - محمد ناصر - محمد محمود محمد محمود - ملك محمد عثمان - عائشة نجاح - إخلاص صبحي
        <br>
        202203664 - 202206090 - 202200570 - 202203785 - 202202236 - 202202554
    </h4>
""", unsafe_allow_html=True)
