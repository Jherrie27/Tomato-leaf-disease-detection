import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image

# Streamlit Page Configuration
st.set_page_config(page_title="Tomato Leaf Disease Detection", layout="wide")

# Sidebar Info
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

# App Title
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
        # Run inference
        results = model(temp_path)

        # Load the original image
        img = cv2.imread(temp_path)

        # Process detections
        for r in results:
            # Draw boxes and labels
            if r.boxes is not None:
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

        # Convert to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Detection Result", use_container_width=True)

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
