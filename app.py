import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page configuration
st.set_page_config(page_title="Tomato Leaf Disease Detection", layout="wide")

# Sidebar content
st.sidebar.title("Tomato Leaf Disease Detection")
st.sidebar.write("**Group Name:** Nold Arn")
st.sidebar.write("**Institution:** Mapúa University")
st.sidebar.markdown("---")
st.sidebar.write("Upload an image of a tomato leaf to detect disease using YOLOv12.")

# Load YOLO model (cached for performance)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Make sure best.pt is in the same folder as this script
    return model

model = load_model()

# Main UI
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image below to analyze its condition.")

def run_detection(image):
    results = model.predict(source=image, save=False, imgsz=640, conf=0.5)
    result_image = results[0].plot()

    # Show detection image
    st.image(result_image, caption="Detection Result", use_container_width=True)

    # Show detection results (disease names + confidence)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("Detected Diseases:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls] if cls in model.names else "Unknown"
            st.write(f"🩺 **{label}** — Confidence: {conf:.2f}")
    else:
        st.info("No diseases detected. The leaf might be healthy or unclear in the image.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Running detection...")
    run_detection(image)
