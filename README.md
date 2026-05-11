# 😷 Face Mask Detector

Real-time face mask detection using Deep Learning and OpenCV.

## 🎯 What it does
Detects whether a person is wearing a face mask or not using your webcam in real-time.

- ✅ Mask ON → Green box
- ❌ No Mask → Red box
- 📊 98% Accuracy

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- OpenCV
- NumPy

## 📁 Project Structure
face_mask_project/
├── face_detector/         ← Face detection model
├── detect_mask.py         ← Real-time webcam detection
└── mask_detector.h5       ← Trained AI model
## 🚀 How to Run
```bash
pip install tensorflow opencv-python
python detect_mask.py
```

## 📊 Model Performance
| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| With Mask    | 0.97      | 0.99   | 0.98     |
| Without Mask | 0.99      | 0.97   | 0.98     |
| **Overall**  | **0.98**  | **0.98** | **0.98** |

## 👨‍💻 Author
Mahabubul Hasan — [@konyhasan-cyber](https://github.com/konyhasan-cyber)
