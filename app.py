import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Import OpenCV safely
try:
    import cv2
except ImportError:
    st.error("⚠️ OpenCV failed to load. Make sure only 'opencv-python-headless' is in requirements.txt.")
    st.stop()

from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
import ultralytics.nn.tasks as tasks
import torch.nn as nn


# ✅ PyTorch 2.6 fix for YOLOv12 models
torch.serialization.add_safe_globals([
    tasks.SegmentationModel,
    tasks.DetectionModel,
    nn.Sequential,
])

# Streamlit UI setup
st.set_page_config(page_title="Tomato Leaf Disease Detection", layout="wide")

st.sidebar.title("Tomato Leaf Disease Detection")
st.sidebar.write("**Group Name:** Nold Arn")
st.sidebar.write("**Institution:** Mapúa University")
st.sidebar.markdown("---")
st.sidebar.write("Upload an image of a tomato leaf to detect disease using YOLOv12.")

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load YOLO model: {e}")
        st.stop()

model = load_model()

# UI
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

        # YOLO inference
        results = model.predict(resized_img, imgsz=640, conf=0.4)

        # Annotated segmentation result
        seg_annotated = results[0].plot(boxes=False)  # segmentation only
        seg_annotated = cv2.cvtColor(seg_annotated, cv2.COLOR_BGR2RGB)
        seg_annotated = cv2.resize(seg_annotated, (img_np.shape[1], img_np.shape[0]))

        # Draw bounding boxes and labels on top
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                label = f"{model.names[cls_id]} ({conf:.2f})"
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(seg_annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(seg_annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st.image(seg_annotated, caption="Detection + Segmentation Result", use_container_width=True)

        # List detected diseases
        detected_classes = [model.names[int(box.cls.cpu().numpy())] for box in results[0].boxes] if results[0].boxes else []
        if detected_classes:
            st.subheader("🩺 Detected Diseases:")
            for cls in set(detected_classes):
                st.write(f"- {cls}")
        else:
            st.info("No diseases detected in this image.")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Footer
st.markdown("""
---
## 📚 References & Acknowledgments
**Dataset:**  
Bhonde, K. (2020). *Tomato Leaf Disease Dataset.* [Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)  

**Framework:**  
Ultralytics YOLOv12 — [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

**Instructor:** Dr. Lysa V. Comia  
**Student Work:** Web deployment, UI/UX design, documentation  

---
""")
