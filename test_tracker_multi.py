# test_tracker_multi.py
import cv2
import pickle
import os
import numpy as np
import mediapipe as mp
from collections import OrderedDict

# System Module Imports
from quality_gate import QualityGate
from face_matcher import AdaptiveFaceMatcher
from event_state_machine import AttendanceStateMachine
from feature_extractor import DeepFeatureExtractor
from anti_spoof import PresentationAttackDetector
from db_manager import DatabaseManager

GALLERY_PATH = "gallery_v2.pkl"
MATCH_THRESHOLD = 0.65
MAX_DISTANCE_METERS = 2.0

# Single Initialization of DB & Google Drive CSV Logger
db_manager = DatabaseManager(
    db_path="attendance.db",
    csv_dir=r"G:\My Drive\Attendance_Logs"
)


class CentroidTracker:
    """
    Persistent Object Tracker based on Euclidean Centroid Distance.
    Maintains consistent track IDs across consecutive camera frames.
    """
    def __init__(self, max_disappeared=10, max_distance_px=120):
        self.next_object_id = 1
        self.objects = OrderedDict()       # track_id -> (cx, cy)
        self.bboxes = OrderedDict()        # track_id -> (x, y, w, h)
        self.disappeared = OrderedDict()   # track_id -> frames_missing
        self.max_disappeared = max_disappeared
        self.max_distance_px = max_distance_px

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.bboxes

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Distance matrix between existing objects and incoming centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance_px:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = rects[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col])

        return self.bboxes


def load_gallery():
    if os.path.exists(GALLERY_PATH):
        with open(GALLERY_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


print("[INIT] Starting Multi-Subject Ocular AI TAS v2 Engine...")
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)
cap = cv2.VideoCapture(0)

quality_gate = QualityGate(blur_threshold=10.0, max_roll_deg=25.0)
matcher = AdaptiveFaceMatcher(match_threshold=MATCH_THRESHOLD)
extractor = DeepFeatureExtractor()

pad_detector = PresentationAttackDetector(fft_grid_threshold=52.0)
centroid_tracker = CentroidTracker(max_disappeared=10, max_distance_px=150)

# Isolated State Machine Registry per Track ID
state_machines = {}

gallery = load_gallery()
print(f"[DATABASE LOADED] {len(gallery)} user profiles registered in gallery.")
print("[SYSTEM READY] Multi-subject simultaneous verification active.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb_frame)

    raw_boxes = []
    keypoints_map = {}

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x_px, y_px = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h)
            w_px, h_px = int(bbox.width * frame_w), int(bbox.height * frame_h)

            box_key = (max(0, x_px), max(0, y_px), max(1, w_px), max(1, h_px))

            # Distance estimation based on bounding box height ratio
            box_ratio = h_px / float(frame_h + 1e-6)
            est_distance = 0.5 / (box_ratio + 1e-6)

            # Filter faces within valid operational range
            if est_distance <= MAX_DISTANCE_METERS:
                raw_boxes.append(box_key)

                kp = detection.location_data.relative_keypoints
                r_eye = (int(kp[0].x * frame_w), int(kp[0].y * frame_h))
                l_eye = (int(kp[1].x * frame_w), int(kp[1].y * frame_h))
                keypoints_map[box_key] = [r_eye, l_eye]

    # Update centroid tracker with filtered bounding boxes
    active_tracks = centroid_tracker.update(raw_boxes)

    # Clean up state machines for expired/lost track IDs
    current_track_ids = set(active_tracks.keys())
    for tid in list(state_machines.keys()):
        if tid not in current_track_ids:
            del state_machines[tid]

    # Process each active tracked subject independently
    for track_id, box in active_tracks.items():
        bx, by, bw, bh = box

        # Initialize dedicated state machine with db_manager attached
        if track_id not in state_machines:
            state_machines[track_id] = AttendanceStateMachine(
                buffer_size=5,
                consensus_k=3,
                match_threshold=MATCH_THRESHOLD,
                cooldown_sec=8.0,
                db_manager=db_manager
            )

        state_machine = state_machines[track_id]

        # Match closest keypoint set to tracked box center
        box_center = (bx + bw / 2.0, by + bh / 2.0)
        best_kp = None
        min_dist = float("inf")

        for k_box, k_pts in keypoints_map.items():
            kc_x, kc_y = k_box[0] + k_box[2] / 2.0, k_box[1] + k_box[3] / 2.0
            dist_to_box = np.sqrt((box_center[0] - kc_x)**2 + (box_center[1] - kc_y)**2)
            if dist_to_box < min_dist:
                min_dist = dist_to_box
                best_kp = k_pts

        is_valid, quality_msg, aligned_face = quality_gate.inspect_and_align(frame, box, best_kp)

        if is_valid:
            # -----------------------------------------------------------------
            # 1. PRESENTATION ATTACK INSPECTION (RAW Frame Crop)
            # -----------------------------------------------------------------
            rx1, ry1 = max(0, bx), max(0, by)
            rx2, ry2 = min(frame_w, bx + bw), min(frame_h, by + bh)
            raw_crop = frame[ry1:ry2, rx1:rx2]

            is_spoof, spoof_reason = pad_detector.inspect_face_roi(raw_crop)

            if is_spoof:
                state_machine.reset_buffer()
                color = (0, 0, 255)  # Red Box
                status = f"ID #{track_id}: SPOOF REJECTED"
            else:
                # -------------------------------------------------------------
                # 2. FEATURE EMBEDDINGS & MATCHING (Lighting-Corrected Crop)
                # -------------------------------------------------------------
                perio_roi = matcher.extract_periocular_roi(aligned_face)
                occlusion_state = matcher.detect_occlusion_state(aligned_face)

                live_full_emb = extractor.extract_embedding(aligned_face)
                live_perio_emb = extractor.extract_embedding(perio_roi)

                best_candidate = "UNKNOWN"
                highest_score = -1.0
                best_s_full = 0.0
                best_s_perio = 0.0

                if gallery:
                    for uid, profile in gallery.items():
                        s_full = matcher.compute_cosine_similarity(live_full_emb, profile["emb_full"])
                        s_perio = matcher.compute_cosine_similarity(live_perio_emb, profile["emb_perio"])
                        score, _, _, _ = matcher.fuse_scores(s_full, s_perio, occlusion_state)

                        if score > highest_score:
                            highest_score = score
                            best_candidate = uid
                            best_s_full = s_full
                            best_s_perio = s_perio

                best_user = best_candidate if highest_score >= MATCH_THRESHOLD else "UNKNOWN"

                # 3. Update Isolated State Machine (Handles Logging internally upon consensus)
                state, color, raw_status = state_machine.process_frame_score(
                    score=highest_score,
                    occlusion_state=occlusion_state,
                    user_id=best_user,
                    face_roi=aligned_face,
                    score_full=best_s_full,
                    score_perio=best_s_perio
                )
                status = f"ID #{track_id}: {raw_status}"
        else:
            color = (0, 165, 255)  # Orange
            status = f"ID #{track_id}: {quality_msg}"

        # Render Bounding Box and Label per Tracked Subject
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)

        # Header tag background for readability
        cv2.rectangle(frame, (bx, max(0, by - 28)), (bx + bw, max(20, by)), color, -1)
        cv2.putText(frame, status, (bx + 5, max(18, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

    cv2.imshow("Ocular AI TAS v2 - Multi-Subject Tracking & Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()