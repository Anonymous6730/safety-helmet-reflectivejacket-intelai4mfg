import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from ultralytics import YOLO

# === Load YOLOv8 Model ===
@st.cache_resource
def load_model():
    return YOLO("C:/Users/khatr/Projects/Intel_Internship/safety-Helmet-Reflective-Jacket/runs/detect/ppe_detection/weights/best.pt")  # path to your trained YOLO model

model = load_model()

# === UI ===
st.title("🦺 Safety Equipment Detector")
st.write("This app detects **Safety Helmets** and **Reflective Jackets** using a YOLOv8 model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # === Save image temporarily to disk (YOLO requires file path) ===
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    # === Run YOLOv8 Prediction ===
    results = model.predict(source=temp_image_path, conf=0.25, save=False)

    # === Draw Results ===
    result_img = results[0].plot()  # Draw bounding boxes on image

    st.image(result_img, caption="Detected Objects", use_container_width=True)

    # === Show detected class names and confidences ===
    st.markdown("### Detection Results")
    boxes = results[0].boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            st.write(f"**{i+1}. {label}** — Confidence: {conf:.2%}")
    else:
        st.warning("No objects detected.")

