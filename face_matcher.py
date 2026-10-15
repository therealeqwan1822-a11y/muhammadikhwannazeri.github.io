# face_matcher.py
import cv2
import numpy as np


class AdaptiveFaceMatcher:
    """
    Adaptive Face & Periocular Feature Matcher.
    Dynamically balances Full-Face and Periocular embeddings based on occlusion state
    (e.g., face mask detection vs full face visibility).
    """
    def __init__(self, match_threshold=0.65):
        self.match_threshold = match_threshold

    def compute_cosine_similarity(self, emb1, emb2):
        """
        Calculates Cosine Similarity between two 512-dimensional feature vectors.
        """
        if emb1 is None or emb2 is None:
            return 0.0

        vec1 = np.array(emb1, dtype=np.float32).flatten()
        vec2 = np.array(emb2, dtype=np.float32).flatten()

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def detect_occlusion_state(self, face_bgr):
        """
        Detects lower-face masks or partial occlusions based on variance in the lower face.
        Returns: 'FULL_FACE' or 'MASK_OCCLUDED'
        """
        if face_bgr is None or face_bgr.size == 0:
            return "FULL_FACE"

        h, w, _ = face_bgr.shape
        lower_half = face_bgr[int(h * 0.55):, :]

        if lower_half.size == 0:
            return "FULL_FACE"

        gray_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY) if len(lower_half.shape) == 3 else lower_half
        std_dev = np.std(gray_lower)

        # Surgical/cloth masks have low surface texture variance (< 18.0)
        if std_dev < 18.0:
            return "MASK_OCCLUDED"

        return "FULL_FACE"

    def extract_periocular_roi(self, face_bgr):
        """
        Crops top 55% of face bounding box (eyes and eyebrow region).
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr

        h, w, _ = face_bgr.shape
        return face_bgr[0:int(h * 0.55), :]

    def fuse_scores(self, score_full, score_perio, occlusion_state):
        """
        Dynamically fuses full face and periocular scores.
        Returns: (fused_score, mode, score_full, score_perio)
        """
        if occlusion_state == "MASK_OCCLUDED":
            # Rely primarily on periocular region when mask is detected
            fused_score = (0.15 * score_full) + (0.85 * score_perio)
            mode = "PERIOCULAR_FUSED"
        else:
            # Full face visible: prioritize full face embedding (85% full, 15% periocular)
            fused_score = (0.85 * score_full) + (0.15 * score_perio)
            mode = "FULL_FACE_PRIMARY"

        return fused_score, mode, score_full, score_perio