import os, sys

# --- Environment fixes for Streamlit + OpenCV ---
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

# Force OpenCV to use headless version
try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip install opencv-python-headless==4.8.0.76")
    import cv2

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
import ultralytics.nn.tasks as tasks
import torch.nn as nn

# ✅ Allow YOLO model architectures and torch containers to load safely (PyTorch 2.6 fix)
torch.serialization.add_safe_globals([
    tasks.SegmentationModel,
    tasks.DetectionModel,
    nn.Sequential,
])

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Tomato Leaf Disease Detection", layout="wide")

# Sidebar info
st.sidebar.title("Tomato Leaf Disease Detection")
st.sidebar.write("**Group Name:** Nold Arn")
st.sidebar.write("**Institution:** Mapúa University")
st.sidebar.markdown("---")
st.sidebar.write("Upload an image of a tomato leaf to detect disease using YOLOv12.")

# --- Load YOLO model (cached) ---
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load YOLO model: {e}")
        st.stop()

model = load_model()

# --- Main UI ---
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image below to analyze its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    try:
        # Load image
        img = Image.open(temp_path).convert("RGB")
        img_np = np.array(img)
        resized_img = cv2.resize(img_np, (640, 640))

        # Run YOLO prediction
        results = model.predict(resized_img, imgsz=640, conf=0.4)

        # --- Combine Segmentation and Bounding Boxes ---
        annotated_seg = results[0].plot()  # segmentation + bounding boxes (from YOLO)
        annotated_seg = cv2.cvtColor(annotated_seg, cv2.COLOR_BGR2RGB)
        annotated_seg = cv2.resize(annotated_seg, (img_np.shape[1], img_np.shape[0]))

        detected_classes = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                label = f"{model.names[cls_id]} ({conf:.2f})"
                detected_classes.append(model.names[cls_id])

                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Draw rectangle + label on segmentation image
                cv2.rectangle(annotated_seg, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(annotated_seg, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show result
        st.image(annotated_seg, caption="Segmentation + Bounding Boxes Result", use_container_width=True)

        # Display detected disease labels
        if detected_classes:
            st.subheader("🩺 Detected Diseases:")
            for cls in set(detected_classes):
                st.write(f"- {cls}")
        else:
            st.info("No diseases detected in this image.")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# --- Footer / Overview ---
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
This app uses **YOLOv12** for instance segmentation & disease classification.

### Features
- Real-time tomato leaf disease detection  
- Instance segmentation + bounding boxes  
- Confidence scoring and labeling  
- Visual results with overlays  

---

## ⚖️ Legal Disclaimer
This model is provided *“as-is”* for educational purposes.  
Developers make no warranties regarding accuracy or suitability.  
Users must verify all outputs before using them in real-world decisions.

**Intellectual Property:** Training methodology by *Dr. Lysa V. Comia*  
Implementation for *academic evaluation only*.
---
""")
