import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("./best.pt")

model = load_model()

st.title("🦺 Safety Equipment Detector")
st.write("This app detects **Safety Helmets** and **Reflective Jackets** using a YOLOv8 model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    results = model.predict(source=temp_image_path, conf=0.25, save=False)

    result_img = results[0].plot() 

    st.image(result_img, caption="Detected Objects", use_container_width=True)

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

