import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Streamlit Page Configuration
st.set_page_config(page_title="Tomato Leaf Disease Detection", layout="wide")

# Sidebar 
st.sidebar.title("Tomato Leaf Disease Detection")
st.sidebar.write("**Group Name:** Nold Arn")
st.sidebar.write("**Institution:** Mapúa University")
st.sidebar.markdown("---")
st.sidebar.write("Upload an image of a tomato leaf to detect disease using YOLOv12.")

# Load YOLO model (cached)
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load YOLO model: {e}")
        st.stop()

model = load_model()

# Title
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image below to analyze its condition.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Save uploaded file safely
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    try:
        # Load image and convert to RGB
        img = Image.open(temp_path).convert("RGB")
        img_np = np.array(img)

        # Resize for YOLO input (640x640 standard)
        resized_img = cv2.resize(img_np, (640, 640))

        # Run YOLO inference with explicit size
        results = model.predict(resized_img, imgsz=640)

        # Get annotated output and resize back to original size
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated = cv2.resize(annotated, (img_np.shape[1], img_np.shape[0]))

        # Show annotated image
        st.image(annotated, caption="Detection and Segmentation Result", use_container_width=True)

        # Show detected classes
        detected_classes = []
        for r in results:
            if r.boxes is not None:
                detected_classes.extend([model.names[int(c)] for c in r.boxes.cls.cpu().numpy()])

        if detected_classes:
            st.subheader("🩺 Detected Diseases:")
            for cls in set(detected_classes):
                st.write(f"- {cls}")
        else:
            st.info("No diseases detected in this image.")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# App overview
st.markdown("""
---

## 📚 References & Acknowledgments
**Dataset:**  
Bhonde, K. (2020). *Tomato Leaf Disease Dataset.* Kaggle.  
[https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)

**Framework:**  
Ultralytics YOLOv12 — [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

**Instructor:** Dr. Lysa V. Comia  
**Training Methodology:** Provided as course material (IP protected)  
**Student Work:** Web deployment, UI/UX design, documentation  

---

## 🧠 Application Overview
This application uses **YOLOv12** for instance segmentation to detect and classify tomato leaf diseases.

### Features
- Real-time disease detection  
- Instance segmentation (colored mask overlay)  
- Confidence scoring  
- Visual analysis  
- Batch processing support  

### How to use
1. Upload your trained model (`.pt` file)  
2. Adjust detection parameters  
3. Upload tomato leaf images  
4. View results and analysis  

---

## 📊 Model Performance & Transparency Report
**Institution:** Mapúa University  
**Course:** AI 2 (Artificial Intelligence 2)  
**Project Type:** Academic Completion Requirement  
**Model Architecture:** YOLOv12n-seg (Ultralytics)  

---

## ⚖️ Legal Disclaimer
This model is provided *“as-is”* for educational purposes.  
Developers make no warranties regarding accuracy or suitability.  
Users must verify all outputs before using them in real-world decisions.

**Intellectual Property:** Training methodology by *Dr. Lysa V. Comia*.  
Implementation is for *academic evaluation only* and may not be redistributed.

**Compute Resources:** Google Colab (A100 GPU, free tier)
---
""")
