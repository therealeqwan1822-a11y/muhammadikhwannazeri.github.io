# event_state_machine.py
import os
import time
import cv2
from datetime import datetime
from mqtt_controller import HardwareDoorController  # <--- IMPORT HARDWARE CONTROLLER

class AttendanceStateMachine:
    def __init__(self, buffer_size=5, consensus_k=3, match_threshold=0.65, cooldown_sec=8.0, db_manager=None):
        self.buffer_size = buffer_size
        self.consensus_k = consensus_k
        self.match_threshold = match_threshold
        self.cooldown_sec = cooldown_sec
        self.db_manager = db_manager

        # Initialize Hardware MQTT Controller
        self.hardware = HardwareDoorController()

        self.score_buffer = []
        self.user_buffer = []
        self.state = "SEARCHING"
        self.last_log_time = 0
        self.snapshot_dir = "attendance_snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def reset_buffer(self):
        self.score_buffer.clear()
        self.user_buffer.clear()

    def process_frame_score(self, score, occlusion_state, user_id, face_roi, score_full=0.0, score_perio=0.0):
        current_time = time.time()

        # 1. Handle Active Cooldown Window
        if self.state == "COOLDOWN":
            elapsed = current_time - self.last_log_time
            remaining = self.cooldown_sec - elapsed
            if remaining > 0:
                return ("COOLDOWN", (128, 128, 128), f"COOLDOWN ({remaining:.1f}s)")
            else:
                self.state = "SEARCHING"
                self.reset_buffer()

        # 2. Append Frame Results to Rolling Window Buffer
        self.score_buffer.append(score)
        self.user_buffer.append(user_id)
        if len(self.score_buffer) > self.buffer_size:
            self.score_buffer.pop(0)
            self.user_buffer.pop(0)

        # Filter Valid Candidate Matches
        valid_matches = [u for u, s in zip(self.user_buffer, self.score_buffer) if s >= self.match_threshold and u != "UNKNOWN"]

        # 3. Consensus K Met -> Trigger Hardware Unlock, Snapshot & Cloud CSV Write
        if len(valid_matches) >= self.consensus_k:
            target_user = max(set(valid_matches), key=valid_matches.count)

            self.last_log_time = current_time
            self.state = "COOLDOWN"

            # TRIGGER VIRTUAL ESP32 DOOR UNLOCK VIA MQTT
            self.hardware.unlock_door()

            # Save Local Image Snapshot
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_filename = f"{target_user}_{now_str}.jpg"
            snap_path = os.path.join(self.snapshot_dir, snap_filename)
            if face_roi is not None and face_roi.size > 0:
                cv2.imwrite(snap_path, face_roi)

            print(f"[EVENT LOGGED] {target_user} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -> Saved {snap_path}")

            # Direct Write to Google Drive CSV & SQLite
            if self.db_manager is not None:
                self.db_manager.log_attendance(
                    user_id=target_user,
                    score_full=round(score_full, 2),
                    score_perio=round(score_perio, 2),
                    fused_score=round(score, 2),
                    status="Dual-Consensus"
                )

            self.reset_buffer()
            return ("COOLDOWN", (0, 255, 0), f"VERIFIED: {target_user}")

        # 4. Verification in Progress
        if len(self.score_buffer) > 0 and user_id != "UNKNOWN":
            return ("MATCHING", (255, 255, 0), f"VERIFYING ({len(valid_matches)}/{self.consensus_k})")

        return ("SEARCHING", (0, 165, 255), "SEARCHING")