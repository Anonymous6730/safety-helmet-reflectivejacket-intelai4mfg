
# 🦺 Helmet & Reflective Jacket Detection with YOLOv11

This project leverages **YOLOv11** and a **Streamlit web app** to detect **safety helmets** and **reflective jackets** in uploaded images. Designed for improving safety compliance in industrial and construction environments, this system enables real-time or batch detection of personal protective equipment (PPE).

---

## 🚧 Project Features

- 🧠 **YOLOv11-based Object Detection**
- 🖼 **Streamlit GUI** for easy interaction
- 🔍 **Detection of 2 classes**: `helmet`, `jacket`
- 📸 **Upload images** and visualize detections instantly

---

## 📁 Files & Structure

```
project-root/
├── app.py                  # Streamlit web app
├── model_training.ipynb    # Model Training Notebook
├── YOLOv11m.pt             # Trained model
└── data.yaml               # Dataset configuration

```

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Anonymous6730/safety-helmet-reflectivejacket-intelai4mfg
cd safety-helmet-reflectivejacket-intelai4mfg
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```
### Usage:

- Upload a `.jpg`, `.jpeg`, or `.png` image
- Get predictions with bounding boxes and confidence scores
- See list of detected items (e.g., Helmet — 98.5%)

---

## 🗃 Dataset Overview

- **Source**: [Kaggle PPE Dataset](https://www.kaggle.com/datasets/niravnaik/safety-helmet-and-reflective-jacket)
- **Structure**: YOLO format
- **Split**: Training / Validation defined in `data.yaml`

Sample `data.yaml`:

```yaml
train: ../train/images
val: ../valid/images

nc: 2
names: ['helmet', 'jacket']
```

## 🧾 License

This project is open-sourced under the **MIT License**.  
Dataset is publicly available on Kaggle and licensed accordingly.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Kaggle Safety Helmet & Reflective Jacket Dataset](https://www.kaggle.com/datasets/niravnaik/safety-helmet-and-reflective-jacket)
- [Streamlit](https://streamlit.io/)

---
