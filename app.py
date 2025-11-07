import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import ultralytics.nn.tasks as tasks  # ✅ Needed for safe loading

# ✅ Allow YOLOv8/YOLOv12 model classes for PyTorch 2.6+
torch.serialization.add_safe_globals([tasks.SegmentationModel, tasks.DetectionModel])

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
