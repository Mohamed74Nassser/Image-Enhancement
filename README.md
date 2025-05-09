# Image Enhancement Web App

## Overview
This is an interactive web application built using **Streamlit** and **OpenCV** that allows users to apply a variety of image enhancement techniques, including denoising, sharpening, contrast stretching, histogram equalization, and more.

## Features
- Brightness adjustment  
- Contrast stretching  
- Global and CLAHE histogram equalization  
- Gaussian and Median denoising  
- Image sharpening  
- Color correction (saturation adjustment)  
- White balance correction  
- Side-by-side image comparison

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Mohamed74Nassser/Image-Enhancement.git
cd Image-Enhancement
```

### 2. Install Dependencies
We recommend using a virtual environment:
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

## Usage
- Upload any image (`.jpg`, `.jpeg`, `.png`)
- Select an enhancement technique from the dropdown
- Tune parameters using sliders
- View and compare original and processed images side-by-side

## Results
The application displays side-by-side comparisons of the original and enhanced images, allowing users to visually evaluate the effect of each enhancement method.  
Key observations include:

- Brightness Adjustment clearly increases or decreases exposure based on user-defined levels.
- Contrast Stretching enhances visual clarity in low-contrast regions.
- Histogram Equalization & CLAHE improve overall contrast, especially in grayscale images.
- Denoising Filters (Gaussian, Median) effectively reduce visual noise while preserving details.
- Sharpening makes edges and textures more defined.
- Color Correction increases vibrance and saturation.
- White Balance corrects color temperature for more natural appearance.

You can upload your own image and try all techniques interactively.

## Tech Stack
- Python
- Streamlit
- OpenCV
- NumPy
- PIL

## Contributors
- طارق ممدوح — 202202554  
- محمد ناصر — 202202236  
- محمد محمود محمد محمود — 202203785  
- ملك محمد عثمان — 202200570  
- عائشة نجاح — 202206090  
- إخلاص صبحي — 202203664

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, modify, and distribute this software with attribution.

## References
- OpenCV Documentation: https://docs.opencv.org/  
- Histogram Equalization: https://en.wikipedia.org/wiki/Histogram_equalization  
- CLAHE: https://en.wikipedia.org/wiki/Adaptive_histogram_equalization  
- Filtering Techniques: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html  
- Sharpening: https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm  
- HSV Color Correction: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html  
- White Balance: https://photo.stackexchange.com/questions/12434/what-is-white-balance  
- Streamlit Documentation: https://docs.streamlit.io/
