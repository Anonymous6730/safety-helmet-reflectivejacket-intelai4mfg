import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import cv2
from ultralytics import YOLO
import time
from io import BytesIO

st.set_page_config(page_title="PPE Detector", page_icon="🦺", layout="centered")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
This app uses a YOLOv11 model to detect:
- 🪖 Safety Helmets  
- 🦺 Reflective Jackets

---

### 🛡️ Model Disclaimer
This model is trained on a specific dataset and may not perform well on all images.  
For best results:
- Use clear, daylight images  
- Ensure PPE items are fully visible

---
""")


st.markdown("<h1 style='text-align: center;'>🦺 PPE Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect Safety Helmets and Reflective Jackets in images using AI.</p>", unsafe_allow_html=True)

st.markdown("### Follow These Steps:")
st.markdown("""
1. Upload a JPG/PNG image of people working on-site.  
2. The model will automatically detect and highlight PPE items.  
3. Results and confidence levels will be shown.  
4. You can download the image with detections or try again!
""")

st.divider()

st.markdown("### Step 1: Upload Your Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    st.markdown("### Step 2: Processing Image...")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    for percent in range(1, 101, 10):
        progress_bar.progress(percent)
        status_text.text(f"Running detection... {percent}%")
        time.sleep(0.03)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    results = model.predict(source=temp_image_path, conf=0.25, save=False)
    result_img = results[0].plot()
    status_text.text("Detection Complete!")
    progress_bar.empty()

    st.markdown("### Step 3: Detection Results")
    st.image(result_img, caption="Detected PPE Items", use_container_width=True)

    boxes = results[0].boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            st.success(f"**{i+1}. {label}** — Confidence: {conf:.2%}")
    else:
        st.warning("⚠️ No PPE items detected. Try another image.")

    st.markdown("### Step 4: Download or Try Another")
    buf = BytesIO()
    Image.fromarray(result_img).save(buf, format="JPEG")
    byte_im = buf.getvalue()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button("Download Image", data=byte_im, file_name="ppe_result.jpg", mime="image/jpeg")
else:
    st.info("Upload an image above to begin PPE detection.")
    st.image("https://cdn-icons-png.flaticon.com/512/4715/4715620.png", width=120, caption="Awaiting image upload", use_container_width=False)

st.divider()
st.markdown("<p style='text-align: center; font-size: 13px;'>Made with 🧠 and ☕ using Streamlit & YOLOv11</p>", unsafe_allow_html=True)
