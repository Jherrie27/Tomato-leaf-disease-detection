import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# Page config
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
    return YOLO("best.pt")

model = load_model()

# Main UI
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image below to analyze its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Run inference
    results = model(temp_path)

    # Load original image
    img = cv2.imread(temp_path)

    # Process detections
    for r in results:
        # Overlay segmentation masks
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                mask = mask.astype(np.uint8) * 255
                colored_mask = np.zeros_like(img)
                colored_mask[:, :, 2] = mask  # Red overlay
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

        # Draw boxes and labels
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        names = model.names

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[cls_id]} ({score:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Convert and display result
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Detection Result", use_container_width=True)

    # Show summary of detected diseases
    detected_classes = [model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
    if detected_classes:
        st.subheader("🩺 Detected Diseases:")
        for cls in detected_classes:
            st.write(f"- {cls}")
    else:
        st.info("No diseases detected in this image.")
