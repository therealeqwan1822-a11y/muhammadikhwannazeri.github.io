# feature_extractor.py
import cv2
import numpy as np
import os

MODEL_PATH = "mobilefacenet.onnx"

class DeepFeatureExtractor:
    """
    Deep Neural Feature Extractor for Ocular AI TAS v2.
    Loads ONNX deep learning embeddings with robust fallback support.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.net = None

        if os.path.exists(self.model_path):
            self._load_onnx()
        else:
            print(f"[INFO] ONNX model not found locally at '{self.model_path}'.")
            print("[INFO] Operating with zero-mean structural feature extractor.")

    def _load_onnx(self):
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[MODEL LOADED] Successfully initialized ONNX network: {self.model_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load ONNX model: {e}")
            self.net = None

    def extract_embedding(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return np.zeros((128,), dtype=np.float32)

        if self.net is not None:
            # Deep Learning Preprocessing (112x112 RGB Blob Normalized)
            blob = cv2.dnn.blobFromImage(
                face_roi,
                scalefactor=1.0 / 127.5,
                size=(112, 112),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
                crop=False
            )
            self.net.setInput(blob)
            raw_emb = self.net.forward().flatten().astype(np.float32)
        else:
            # Enhanced Structural Gradient Descriptor
            resized = cv2.resize(face_roi, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized.copy()

            # Explicit float32 casting to avoid float64 OpenCV SIMD format errors
            mean_val = float(np.mean(gray))
            std_val = float(np.std(gray))
            gray_float = (gray.astype(np.float32) - mean_val) / (std_val + 1e-6)
            gray_float = gray_float.astype(np.float32)

            gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)

            raw_emb = np.hstack([gray_float.flatten(), mag.flatten()])

        # L2 Vector Normalization
        norm = np.linalg.norm(raw_emb)
        return raw_emb / (norm + 1e-6)