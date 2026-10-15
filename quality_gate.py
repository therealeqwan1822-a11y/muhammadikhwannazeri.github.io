# quality_gate.py
import cv2
import numpy as np

class QualityGate:
    """
    Quality Inspection & Lighting Adaptation Engine.
    Applies linear YUV luminance scaling to preserve embedding feature vectors 
    for deep face recognition models.
    """
    def __init__(self, blur_threshold=12.0, max_roll_deg=25.0):
        self.blur_threshold = blur_threshold
        self.max_roll_deg = max_roll_deg

    def adapt_lighting_linear(self, face_bgr):
        """
        Global linear intensity scaling in YUV color space.
        Normalizes dark or backlit faces without introducing local contrast artifacts.
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr

        # Convert BGR to YUV space (Y = Luminance)
        yuv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YUV)
        y_channel, u_channel, v_channel = cv2.split(yuv)

        mean_y = np.mean(y_channel)

        # Apply scaling if frame is under- or over-exposed
        if mean_y < 85.0 or mean_y > 175.0:
            target_y = 125.0
            gain = np.clip(target_y / (mean_y + 1e-5), 0.6, 2.0)
            
            # Linearly scale luminance channel
            scaled_y = cv2.addWeighted(y_channel, gain, 0, 0, 0)
            yuv_scaled = cv2.merge((scaled_y, u_channel, v_channel))
            return cv2.cvtColor(yuv_scaled, cv2.COLOR_YUV2BGR)

        return face_bgr

    def inspect_and_align(self, frame, bbox, keypoints=None):
        """
        Performs pose checking, blur validation, and exposure adjustment.
        Returns: (is_valid: bool, reason: str, processed_face: np.ndarray)
        """
        h_frame, w_frame, _ = frame.shape
        x, y, w, h = bbox

        # Pad bounding box by 10%
        pad_x, pad_y = int(w * 0.10), int(h * 0.10)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w_frame, x + w + pad_x), min(h_frame, y + h + pad_y)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
            return False, "FACE_TOO_SMALL", None

        # 1. Roll Angle Check
        aligned_face = face_crop
        if keypoints is not None and len(keypoints) >= 2:
            r_eye, l_eye = keypoints[0], keypoints[1]
            dx, dy = l_eye[0] - r_eye[0], l_eye[1] - r_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))

            if abs(angle) > self.max_roll_deg:
                return False, f"HEAD_ROLL_EXCEEDED ({angle:.1f}deg)", None

            eye_center = (int((r_eye[0] + l_eye[0]) / 2.0 - x1), int((r_eye[1] + l_eye[1]) / 2.0 - y1))
            rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            aligned_face = cv2.warpAffine(face_crop, rot_mat, (face_crop.shape[1], face_crop.shape[0]))

        # 2. Blur Check
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < self.blur_threshold:
            return False, f"IMAGE_BLURRY ({blur_score:.1f})", None

        # 3. Apply Linear Lighting Correction
        corrected_face = self.adapt_lighting_linear(aligned_face)

        return True, "QUALITY_PASS", corrected_face