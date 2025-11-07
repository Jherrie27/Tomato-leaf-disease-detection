import os

# Environment fixes for Streamlit + OpenCV
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

import streamlit as st
from ultralytics import YOLO
import cv2
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
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Save uploaded file safely
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    try:
        # Load and prepare image
        img = Image.open(temp_path).convert("RGB")
        img_np = np.array(img)
        resized_img = cv2.resize(img_np, (640, 640))

        # Run YOLO inference
        results = model.predict(resized_img, imgsz=640, conf=0.4)

        names = model.names
        detected_classes = []
        output_img = resized_img.copy()

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    label = names[cls_id] if cls_id in names else f"Class {cls_id}"
                    conf = float(box.conf[0])
                    detected_classes.append(label)

                    # --- Draw bounding box ---
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(output_img, f"{label} ({conf:.2f})", (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # --- Draw segmentation mask if available ---
                    if masks is not None and len(masks.data) > i:
                        mask = masks.data[i].cpu().numpy()
                        colored_mask = np.zeros_like(output_img, dtype=np.uint8)
                        colored_mask[mask > 0.5] = [0, 255, 0]  # green mask
                        output_img = cv2.addWeighted(output_img, 1, colored_mask, 0.4, 0)

        # Resize back to original for display
        output_img = cv2.resize(output_img, (img_np.shape[1], img_np.shape[0]))
        st.image(output_img, caption="Detection and Segmentation Result", use_container_width=True)

        # Display detected diseases
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
- Bounding boxes and class labels  
- Confidence scoring  
- Visual analysis  

---

## ⚖️ Legal Disclaimer
This model is provided *“as-is”* for educational purposes.  
Developers make no warranties regarding accuracy or suitability.  
Users must verify all outputs before using them in real-world decisions.

**Intellectual Property:** Training methodology by *Dr. Lysa V. Comia*.  
Implementation is for *academic evaluation only* and may not be redistributed.
---
""")
