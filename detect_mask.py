import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

print("[INFO] মডেল লোড হচ্ছে...")
maskNet = load_model("mask_detector.h5")
faceNet = cv2.dnn.readNet(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces, locs, preds = [], [], []

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
            faces.append(face)
            locs.append((sX, sY, eX, eY))

    if len(faces) > 0:
        preds = maskNet.predict(np.array(faces, dtype="float32"), batch_size=32)
    return (locs, preds)

print("[INFO] Webcam চালু হচ্ছে... (বন্ধ করতে Q চাপো)")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    (locs, preds) = detect_mask(frame)
    for (box, pred) in zip(locs, preds):
        (sX, sY, eX, eY) = box
        (mask, withoutMask) = pred
        if mask > withoutMask:
            label = "Mask ON"; color = (0, 255, 0)
        else:
            label = "No Mask!"; color = (0, 0, 255)
        label = f"{label}: {max(mask, withoutMask)*100:.1f}%"
        cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
        cv2.putText(frame, label, (sX, sY-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()