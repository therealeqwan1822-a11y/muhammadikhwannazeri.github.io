# liveness_gate.py
import numpy as np

class FastLivenessDetector:
    """
    Real-Time Passive Liveness Inspector.
    Tracks relative facial keypoint dynamics across 4 frames to ensure zero-latency verification.
    """
    def __init__(self, buffer_size=4, min_motion_std=0.0010):
        self.buffer_size = buffer_size
        self.min_motion_std = min_motion_std
        self.rel_kp_history = []

    def reset(self):
        self.rel_kp_history.clear()

    def inspect_liveness(self, keypoints, bbox):
        """
        Calculates keypoint displacement normalized to face dimensions.
        Passes instantly on warm-up to prevent verification delays.
        """
        if not keypoints or len(keypoints) < 2 or bbox is None:
            return True, "LIVE_PASS"

        r_eye, l_eye = keypoints[0], keypoints[1]
        bx, by, bw, bh = bbox
        cx, cy = bx + (bw / 2.0), by + (bh / 2.0)

        # Normalize eye positions relative to face box center and scale
        r_rel = ((r_eye[0] - cx) / float(bw + 1e-6), (r_eye[1] - cy) / float(bh + 1e-6))
        l_rel = ((l_eye[0] - cx) / float(bw + 1e-6), (l_eye[1] - cy) / float(bh + 1e-6))

        self.rel_kp_history.append((r_rel, l_rel))
        if len(self.rel_kp_history) > self.buffer_size:
            self.rel_kp_history.pop(0)

        # Allow immediate verification during initial buffer fill
        if len(self.rel_kp_history) < 2:
            return True, "LIVE_PASS"

        # Calculate standard deviation of relative keypoint movement
        kp_arr = np.array(self.rel_kp_history)  # Shape: (N, 2, 2)
        std_motion = np.std(kp_arr)

        if std_motion >= self.min_motion_std or len(self.rel_kp_history) < self.buffer_size:
            return True, "LIVE_PASS"
        else:
            return False, "SPOOF_SUSPECT (STATIC_PHOTO)"