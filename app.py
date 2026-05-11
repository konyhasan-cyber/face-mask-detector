import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Face Mask Detector", page_icon="😷")
st.title("😷 Face Mask Detector")
st.write("ছবি তুলুন — AI বলবে মাস্ক আছে কিনা!")

@st.cache_resource
def load_models():
    maskNet = load_model("mask_detector.h5")
    faceNet = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    )
    return maskNet, faceNet

maskNet, faceNet = load_models()

img_file = st.camera_input("📷 Camera চালু করুন")

if img_file:
    bytes_data = img_file.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    result_text = ""
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sX, sY, eX, eY) = box.astype("int")
            (sX, sY) = (max(0, sX), max(0, sY))
            (eX, eY) = (min(w-1, eX), min(h-1, eY))
            face = frame[sY:eY, sX:eX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(img_to_array(face))
            (mask, withoutMask) = maskNet.predict(np.array([face]))[0]
            if mask > withoutMask:
                label = "✅ Mask ON"
                color = (0, 255, 0)
            else:
                label = "❌ No Mask!"
                color = (0, 0, 255)
            confidence = max(mask, withoutMask) * 100
            result_text += f"{label}: {confidence:.1f}%\n"
            cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.1f}%",
                       (sX, sY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result_text:
        st.success(result_text)
    else:
        st.warning("মুখ পাওয়া যায়নি — আরেকটু কাছে আসুন!")