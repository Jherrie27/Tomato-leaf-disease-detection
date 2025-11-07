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
# confidence slider for controlling detection sensitivity
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.40, 0.05)

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
        # Load original image
        img = Image.open(temp_path).convert("RGB")
        img_np = np.array(img)                          # original HxW
        orig_h, orig_w = img_np.shape[:2]

        # Prepare model input size (we use 640)
        model_size = 640
        resized_img = cv2.resize(img_np, (model_size, model_size))

        # Run YOLO inference (segmentation + detection)
        results = model.predict(resized_img, imgsz=model_size, conf=conf_threshold)
        result = results[0]

        # segmentation visualization (from model) at model resolution
        seg_annotated = result.plot(boxes=False)       # segmentation only (640x640)
        seg_annotated = cv2.cvtColor(seg_annotated, cv2.COLOR_BGR2RGB)

        # Resize segmentation visualization to original image size
        seg_annotated = cv2.resize(seg_annotated, (orig_w, orig_h))

        # Debug: show counts (optional, helps troubleshooting)
        boxes_count = len(result.boxes) if result.boxes is not None else 0
        masks_count = len(result.masks.data) if (result.masks is not None and result.masks.data is not None) else 0
        st.write(f"🧪 Debug — boxes: {boxes_count}, masks: {masks_count}, conf: {conf_threshold}")

        # If any boxes exist, draw them — but SCALE coordinates from model-space -> original-space
        if boxes_count > 0:
            # scaling factors
            scale_x = orig_w / model_size
            scale_y = orig_h / model_size

            for box in result.boxes:
                # box.xyxy is in model-space (640)
                xy = box.xyxy[0].cpu().numpy().astype(float)  # [x1, y1, x2, y2] at model scale
                x1_model, y1_model, x2_model, y2_model = xy
                # convert to original image coords
                x1 = int(round(x1_model * scale_x))
                y1 = int(round(y1_model * scale_y))
                x2 = int(round(x2_model * scale_x))
                y2 = int(round(y2_model * scale_y))

                cls_id = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                label = f"{model.names[cls_id]} ({conf:.2f})"

                # draw rectangle and label on seg_annotated (which is original size now)
                cv2.rectangle(seg_annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(seg_annotated, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # fallback: if boxes empty but masks exist, derive bboxes from masks (and label as Unknown)
            if masks_count > 0:
                # result.masks.data are at model scale; iterate and compute bbox, then scale
                for i, mask_tensor in enumerate(result.masks.data):
                    mask_np = mask_tensor.cpu().numpy()  # model_size x model_size
                    ys, xs = np.where(mask_np > 0.5)
                    if ys.size == 0 or xs.size == 0:
                        continue
                    x1_model, x2_model = xs.min(), xs.max()
                    y1_model, y2_model = ys.min(), ys.max()
                    # scale to original
                    x1 = int(round(x1_model * (orig_w / model_size)))
                    y1 = int(round(y1_model * (orig_h / model_size)))
                    x2 = int(round(x2_model * (orig_w / model_size)))
                    y2 = int(round(y2_model * (orig_h / model_size)))

                    # label fallback - if you want class names but boxes are empty there's no class id.
                    # We'll try to read class id from result.boxes (not present), so fallback to "Detected"
                    label = "Detected (mask)"

                    cv2.rectangle(seg_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(seg_annotated, label, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display combined result
        st.image(seg_annotated, caption="Detection + Segmentation Result", use_container_width=True)

        # Show detected diseases (prefer names from boxes; if empty, try fallback)
        detected_classes = []
        if boxes_count > 0:
            detected_classes = [model.names[int(box.cls.cpu().numpy())] for box in result.boxes]
        elif masks_count > 0:
            # no class ids available from boxes; show generic message
            detected_classes = ["(mask detected)"]

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
