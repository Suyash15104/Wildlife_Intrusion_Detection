import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Elephant Detection", layout="wide")
st.title("🐘 Elephant & Human Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

ELEPHANT_CONF = 0.45
HUMAN_CONF = 0.6

def process_frame(frame):
    results = model(frame, verbose=False)
    elephant_found = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "elephant" and conf >= ELEPHANT_CONF:
                elephant_found = True
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

            elif label == "person" and conf >= HUMAN_CONF:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)

    if elephant_found:
        cv2.putText(frame, "ELEPHANT DETECTED",
                    (30,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)

    return frame


uploaded_file = st.file_uploader("Upload video", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        stframe.image(frame, channels="BGR")

    cap.release()
