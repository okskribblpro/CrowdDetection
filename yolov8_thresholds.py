import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os

# Fix for PyTorch + Streamlit file watcher issue
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("👥 People Counter with YOLOv8")

# Threshold sliders
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy array
    img_array = np.array(image)

    # Run detection with custom thresholds
    results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold)
    result = results[0]

    # Get boxes and classes
    boxes = result.boxes
    person_count = 0

    # Draw only person boxes (class 0)
    for box, cls in zip(boxes.xyxy, boxes.cls):
        if int(cls) == 0:
            person_count += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.markdown(f"### 👤 People detected: {person_count}")

    # Convert to RGB for Streamlit
    annotated_image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption="Detected People", use_container_width=True)
