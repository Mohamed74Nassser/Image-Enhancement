import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set Streamlit layout to wide for better image display
st.set_page_config(layout="wide")

# Adjust image brightness by adding a constant value (beta)
def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)
