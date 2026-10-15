# enroll_v2.py
import cv2
import pickle
import os
import time
import numpy as np
import mediapipe as mp
from quality_gate import QualityGate
from face_matcher import AdaptiveFaceMatcher
from feature_extractor import DeepFeatureExtractor

GALLERY_PATH = "gallery_v2.pkl"
SCAN_DURATION_SEC = 10.0  # 10-second scanning window
MIN_REQUIRED_SAMPLES = 20  # Minimum valid frames captured to save profile

def load_gallery():
    if os.path.exists(GALLERY_PATH):
        with open(GALLERY_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_gallery(gallery):
    with open(GALLERY_PATH, 'wb') as f:
        pickle.dump(gallery, f)
    print(f"\n[GALLERY SAVED] Persistent database updated at {GALLERY_PATH}")

print("[INIT] Loading Vision & Feature Engines...")
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
quality_gate = QualityGate(blur_threshold=10.0, max_roll_deg=25.0)  # Relaxed roll angle for dynamic scans
matcher = AdaptiveFaceMatcher()
extractor = DeepFeatureExtractor()

# Flush TFLite initialization output
mp_face.process(np.zeros((100, 100, 3), dtype=np.uint8))
print("[INIT COMPLETE] System Ready.\n")

user_id = input("Enter User ID / Name for Enrollment (e.g., EMP_001): ").strip()
if not user_id:
    print("User ID cannot be empty.")
    exit()

cap = cv2.VideoCapture(0)

samples_full = []
samples_perio = []

scanning_started = False
start_time = None

print(f"\n[INSTRUCTIONS] Position your face in front of the camera.")
print(f"Once detected, a 10-second scan will begin.")
print(f"Follow on-screen prompts to slowly tilt and turn your head.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb_frame)

    # UI Default States
    status_msg = "ALIGN FACE TO START 10s SCAN"
    guide_msg = "Position face centrally"
    box_color = (255, 255, 0)
    progress = 0.0

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        bx, by = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h)
        bw, bh = int(bbox.width * frame_w), int(bbox.height * frame_h)

        kp = detection.location_data.relative_keypoints
        r_eye = (int(kp[0].x * frame_w), int(kp[0].y * frame_h))
        l_eye = (int(kp[1].x * frame_w), int(kp[1].y * frame_h))

        box = (max(0, bx), max(0, by), max(1, bw), max(1, bh))
        is_valid, q_msg, aligned_face = quality_gate.inspect_and_align(frame, box, [r_eye, l_eye])

        # Start 10s Timer on First Valid Quality Detection
        if is_valid and not scanning_started:
            scanning_started = True
            start_time = time.time()

        if scanning_started:
            elapsed = time.time() - start_time
            remaining = max(0.0, SCAN_DURATION_SEC - elapsed)
            progress = min(1.0, elapsed / SCAN_DURATION_SEC)

            # Dynamic Head Movement Prompts based on Elapsed Time
            if elapsed < 3.0:
                guide_msg = "Look Straight at Camera"
            elif elapsed < 6.0:
                guide_msg = "Slowly Turn Head Left & Right"
            elif elapsed < 8.5:
                guide_msg = "Slowly Tilt Head Up & Down"
            else:
                guide_msg = "Hold Still - Finalizing Scan..."

            if remaining > 0:
                if is_valid:
                    box_color = (0, 255, 0)
                    status_msg = f"SCANNING... {remaining:.1f}s LEFT ({len(samples_full)} frames)"

                    # Non-blocking continuous feature collection
                    perio_roi = matcher.extract_periocular_roi(aligned_face)
                    emb_full = extractor.extract_embedding(aligned_face)
                    emb_perio = extractor.extract_embedding(perio_roi)

                    samples_full.append(emb_full)
                    samples_perio.append(emb_perio)
                else:
                    box_color = (0, 165, 255)
                    status_msg = f"HOLD STILL: {q_msg} ({remaining:.1f}s)"
            else:
                # 10 Seconds Completed
                break

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 2)

    # Render HUD Interface
    # 1. Overlay Header Box
    cv2.rectangle(frame, (0, 0), (frame_w, 90), (20, 20, 20), -1)

    # 2. Status & Guidance Text
    cv2.putText(frame, status_msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"PROMPT: {guide_msg}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    # 3. Progress Bar (Green line filling across frame)
    if scanning_started:
        bar_w = int(frame_w * progress)
        cv2.rectangle(frame, (0, 85), (bar_w, 90), (0, 255, 0), -1)

    cv2.imshow("Ocular AI TAS v2 - 10s Dynamic Face Mapping", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[CANCELLED] Enrollment aborted by user.")
        break

cap.release()
cv2.destroyAllWindows()

# Aggregation & Processing
if len(samples_full) >= MIN_REQUIRED_SAMPLES:
    print(f"\n[PROCESSING] Aggregating {len(samples_full)} multi-pose feature maps...")

    # Compute centroid mean embedding vector across all captured poses
    mean_full = np.mean(samples_full, axis=0)
    norm_full = mean_full / (np.linalg.norm(mean_full) + 1e-6)

    mean_perio = np.mean(samples_perio, axis=0)
    norm_perio = mean_perio / (np.linalg.norm(mean_perio) + 1e-6)

    gallery = load_gallery()
    gallery[user_id] = {
        "emb_full": norm_full,
        "emb_perio": norm_perio,
        "sample_count": len(samples_full),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_gallery(gallery)
    print(f"[SUCCESS] Multi-pose 3D vector map for '{user_id}' generated successfully!")
    print(f"[PROFILE DETAILS] Saved with {len(samples_full)} sampled face poses.")
else:
    print(f"\n[FAILED] Insufficient valid frames captured ({len(samples_full)}/{MIN_REQUIRED_SAMPLES}). Please re-run enrollment.")