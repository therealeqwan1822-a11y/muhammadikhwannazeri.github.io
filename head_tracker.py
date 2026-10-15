import cv2
import numpy as np

class KalmanBoxTracker:
    def __init__(self, box):
        # State: [cx, cy, vx, vy, w, h]
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        self.kf.statePost = np.array([[cx], [cy], [0], [0], [w], [h]], dtype=np.float32)
        self.time_since_update = 0

    def get_smoothed_box(self):
        cx, cy, _, _, w, h = self.kf.statePost.ravel()
        x = int(cx - w / 2.0)
        y = int(cy - h / 2.0)
        return [max(0, x), max(0, y), max(1, int(w)), max(1, int(h))]

    def predict(self):
        prediction = self.kf.predict()
        cx, cy, _, _, w, h = prediction.ravel()
        x = int(cx - w / 2.0)
        y = int(cy - h / 2.0)
        self.time_since_update += 1
        return [max(0, x), max(0, y), max(1, int(w)), max(1, int(h))]

    def update(self, box):
        self.time_since_update = 0
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        return self.get_smoothed_box()


class DistanceGatedTracker:
    def __init__(self, max_distance_m=2.0, max_missing_frames=15):
        self.max_distance_m = max_distance_m
        self.head_height_m = 0.22  # Average human head height in meters
        self.max_missing_frames = max_missing_frames
        self.tracker = None

    def _compute_distance(self, box_h_px, frame_height_px):
        if box_h_px <= 0 or frame_height_px <= 0:
            return float('inf')
        # Dynamically scale focal length based on active camera resolution height
        focal_length_px = frame_height_px * 1.065
        return (self.head_height_m * focal_length_px) / box_h_px

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def process_frame(self, detected_boxes, frame_height_px):
        best_det = None
        if detected_boxes:
            best_det = max(detected_boxes, key=lambda b: b[2] * b[3])

        predicted_box = None
        if self.tracker is not None:
            predicted_box = self.tracker.predict()

        if best_det is not None:
            if self.tracker is None:
                self.tracker = KalmanBoxTracker(best_det)
                active_box = best_det
            else:
                iou_score = self._iou(predicted_box, best_det)
                if iou_score > 0.15:
                    active_box = self.tracker.update(best_det)
                else:
                    self.tracker = KalmanBoxTracker(best_det)
                    active_box = best_det
        else:
            if self.tracker is not None and self.tracker.time_since_update <= self.max_missing_frames:
                active_box = predicted_box
            else:
                self.tracker = None
                return None, float('inf'), False

        distance_m = self._compute_distance(active_box[3], frame_height_px)
        trigger_match = distance_m <= self.max_distance_m

        return active_box, distance_m, trigger_match