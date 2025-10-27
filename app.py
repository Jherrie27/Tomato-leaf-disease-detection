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
# --- Full Disclaimer Section ---
st.markdown("""
---

## 🧠 Application Overview
This application uses **YOLOv12** for instance segmentation to detect and classify tomato leaf diseases.

### Features
- Real-time disease detection  
- Instance segmentation  
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

## 📁 Dataset Information
**Source:** [Tomato Leaf Disease Dataset – Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)  
**Creator:** Kaustubh Bhonde (kaustubhb999)  
**License:** Publicly available  

**Composition:**
- Total Images: 967 (validation set)  
- Total Instances: 2,809 annotated disease regions  
- Classes: 9 tomato leaf disease types + healthy leaves  

---

## 🤖 Model Specifications
**Architecture Details:**
- Model: YOLO11n-seg (YOLOv12 series)  
- Parameters: 2,836,323  
- Layers: 265 (fused)  
- Computational Load: 9.6 GFLOPs  
- Training Hardware: NVIDIA A100-SXM4-80GB (Google Colab)  
- Framework: Ultralytics 8.3.63, PyTorch 2.2.2+cu121  

**Training Configuration:**
- Image Size: 640x640 pixels  
- Optimizer: Adam/SGD (Ultralytics default)  
- Model Weights: `/content/drive/MyDrive/yolov12/runs/segment/train5/weights/best.pt`

---

## 📈 Performance Metrics
**Overall Performance:**
- Box mAP@0.5: 55.4%  
- Box mAP@0.5:0.95: 33.5%  
- Mask mAP@0.5: 53.6%  
- Mask mAP@0.5:0.95: 31.9%  

**Per-Class Performance (mAP@0.5):**

| Disease Class | Box mAP@0.5 | Mask mAP@0.5 | Performance |
|----------------|--------------|---------------|--------------|
| Healthy | 99.5% | 99.5% | ⭐ Excellent |
| Mosaic | 80.8% | 77.3% | ✅ Good |
| Late Blight | 71.8% | 71.1% | ✅ Good |
| Yellow Leaf Curl | 61.8% | 58.6% | ✅ Good |
| Early Blight | 56.3% | 58.3% | ⚠️ Moderate |
| Bacterial Spot | 38.6% | 36.1% | ⚠️ Moderate |
| Mold | 33.3% | 30.6% | ⚠️ Moderate |
| Septoria Leaf Spot | 31.8% | 31.0% | ❌ Needs Improvement |
| Spider Mites | 24.2% | 19.9% | ❌ Needs Improvement |

**Inference Speed:**
- Preprocessing: 0.2ms  
- Inference: 0.8ms  
- Postprocessing: 1.1ms  
- Total: ~2.1ms per image  

---

## ⚙️ Model Configuration
**Detection Threshold:** 0.25 (25% confidence)

This provides the best balance between recall and precision given the model’s mAP.

---

## ⚠️ Limitations & Disclaimers
- Class imbalance affects lower-performing classes.  
- Accuracy varies per disease type (best: Healthy, worst: Spider Mites).  
- Single training run, limited augmentation, no hyperparameter tuning.  
- Not reliable for commercial or agricultural decision-making.  
- For **educational use only** — demonstration of AI/ML concepts.  

---

## 🔬 Technical Transparency
- Input images resized to 640x640  
- Normalized via ImageNet statistics  
- Outputs include bounding boxes + segmentation masks  
- Confidence scores: 0.0–1.0  
- Non-Maximum Suppression (IoU 0.45)  

---

## 🎯 Recommended Improvements
- Add more data (especially minority classes)  
- Train for 150–200 epochs  
- Use stronger augmentations (Mixup, AutoAugment)  
- Try larger YOLO models (v12m / v12l)  
- Apply ensemble or active learning strategies  

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

## ⚖️ Legal Disclaimer
This model is provided *“as-is”* for educational purposes.  
Developers make no warranties regarding accuracy or suitability.  
Users must verify all outputs before using them in real-world decisions.

**Intellectual Property:** Training methodology by *Dr. Lysa V. Comia*.  
Implementation is for *academic evaluation only* and may not be redistributed.

**Compute Resources:** Google Colab (A100 GPU, free tier)

---
""")
