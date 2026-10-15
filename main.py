# main.py
import cv2
import json
import mediapipe as mp
import numpy as np
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime

# System Module Imports
from anti_spoof import PresentationAttackDetector
from db_manager import DatabaseManager
from face_matcher import AdaptiveFaceMatcher
from feature_extractor import DeepFeatureExtractor
from quality_gate import QualityGate

GALLERY_PATH = "gallery_v2.pkl"
CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "MATCH_THRESHOLD": 0.65,
    "MAX_DISTANCE_METERS": 2.0,
    "CONSENSUS_K": 3,
    "BUFFER_SIZE": 5,
    "COOLDOWN_SEC": 8.0,
    "BLUR_THRESHOLD": 10.0,
    "MAX_ROLL_DEG": 25.0,
    "FFT_GRID_THRESHOLD": 52.0,
    "CSV_DIR": r"G:\My Drive\Attendance_Logs",
    "DB_PATH": "attendance.db",
}


def load_config():
  if os.path.exists(CONFIG_PATH):
    try:
      with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
          config.setdefault(k, v)
        return config
    except Exception as e:
      print(f"[CONFIG WARNING] Failed to load config.json: {e}")
  return DEFAULT_CONFIG.copy()


def save_config(config):
  try:
    with open(CONFIG_PATH, "w") as f:
      json.dump(config, f, indent=4)
    print("[CONFIG SUCCESS] Settings saved to config.json")
  except Exception as e:
    print(f"[CONFIG ERROR] Failed to save config: {e}")


def clear_screen():
  os.system("cls" if os.name == "nt" else "clear")


def display_menu():
  clear_screen()
  print("=====================================================")
  print("      OCULAR AI TAS - MAIN CONTROL CENTER            ")
  print("=====================================================")
  print(
      "[1]  Launch Live Attendance Engine "
  )
  print("[2]  User Enrollment & Gallery Management")
  print("[3]  Attendance Logs & Reports")
  print("[4]  System Diagnostics & Calibration")
  print("[5]  Settings & Threshold Configuration")
  print("[6]  Database & Cloud Sync Maintenance")
  print("[0]  Exit System")
  print("=====================================================")


# -----------------------------------------------------------------------------
# [1] LIVE ATTENDANCE ENGINE
# -----------------------------------------------------------------------------
def launch_live_engine():
  print("\n[INFO] Starting Ocular AI Attendance Engine...")
  print("[INFO] Press 'q' inside the video window to stop and return to menu.\n")
  try:
    subprocess.run([sys.executable, "test_tracker_multi.py"], check=True)
  except Exception as e:
    print(f"[ERROR] Engine process ended: {e}")
  input("\nPress Enter to return to main menu...")


# -----------------------------------------------------------------------------
# [2] USER ENROLLMENT & GALLERY MANAGEMENT
# -----------------------------------------------------------------------------
def load_gallery():
  if os.path.exists(GALLERY_PATH):
    with open(GALLERY_PATH, "rb") as f:
      return pickle.load(f)
  return {}


def save_gallery(gallery):
  with open(GALLERY_PATH, "wb") as f:
    pickle.dump(gallery, f)


def enroll_new_user():
  clear_screen()
  print("=====================================================")
  print("               USER ENROLLMENT WIZARD                ")
  print("=====================================================")

  user_id = input("Enter User ID / Name to enroll: ").strip()
  if not user_id:
    print("[CANCELLED] User ID cannot be empty.")
    input("\nPress Enter to return...")
    return

  gallery = load_gallery()
  if user_id in gallery:
    overwrite = (
        input(f"'{user_id}' already exists. Overwrite profile? (y/n): ")
        .strip()
        .lower()
    )
    if overwrite != "y":
      print("[CANCELLED] Enrollment aborted.")
      input("\nPress Enter to return...")
      return

  config = load_config()
  quality_gate = QualityGate(
      blur_threshold=config["BLUR_THRESHOLD"],
      max_roll_deg=config["MAX_ROLL_DEG"],
  )
  extractor = DeepFeatureExtractor()
  matcher = AdaptiveFaceMatcher(match_threshold=config["MATCH_THRESHOLD"])

  mp_face = mp.solutions.face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.35
  )
  cap = cv2.VideoCapture(0)

  print("\n[INSTRUCTIONS] Look straight into the camera.")
  print("Align your face until frame is GREEN, then press SPACEBAR to capture.")
  print("Press 'q' to cancel.\n")

  captured = False
  emb_full, emb_perio = None, None

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame_h, frame_w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    status_msg = "No Face Detected"
    color = (0, 165, 255)
    aligned_face = None

    if results.detections:
      detection = results.detections[0]
      bbox = detection.location_data.relative_bounding_box
      x_px, y_px = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h)
      w_px, h_px = int(bbox.width * frame_w), int(bbox.height * frame_h)
      box = (max(0, x_px), max(0, y_px), max(1, w_px), max(1, h_px))

      kp = detection.location_data.relative_keypoints
      r_eye = (int(kp[0].x * frame_w), int(kp[0].y * frame_h))
      l_eye = (int(kp[1].x * frame_w), int(kp[1].y * frame_h))

      is_valid, quality_msg, aligned_face = quality_gate.inspect_and_align(
          frame, box, [r_eye, l_eye]
      )

      if is_valid:
        status_msg = "READY: Press SPACEBAR to capture"
        color = (0, 255, 0)
      else:
        status_msg = f"QUALITY REJECT: {quality_msg}"
        color = (0, 0, 255)

      cv2.rectangle(
          frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2
      )

    cv2.putText(
        frame,
        status_msg,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imshow("Enrollment Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      print("[CANCELLED] Enrollment aborted by user.")
      break
    elif key == 32 and aligned_face is not None:  # Spacebar
      perio_roi = matcher.extract_periocular_roi(aligned_face)
      emb_full = extractor.extract_embedding(aligned_face)
      emb_perio = extractor.extract_embedding(perio_roi)

      gallery[user_id] = {
          "emb_full": emb_full,
          "emb_perio": emb_perio,
          "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      }
      save_gallery(gallery)
      print(f"\n[ENROLLMENT SUCCESS] '{user_id}' registered successfully!")
      captured = True
      break

  cap.release()
  cv2.destroyAllWindows()
  input("\nPress Enter to return to gallery menu...")


def gallery_management():
  while True:
    clear_screen()
    gallery = load_gallery()
    print("=====================================================")
    print("         USER ENROLLMENT & GALLERY MANAGEMENT        ")
    print("=====================================================")
    print(f"Registered User Profiles: {len(gallery)}\n")
    print("[1]  Enroll New User")
    print("[2]  List All Registered Users")
    print("[3]  Delete User Profile")
    print("[0]  Back to Main Menu")
    print("=====================================================")

    choice = input("Select an option [0-3]: ").strip()

    if choice == "1":
      enroll_new_user()
    elif choice == "2":
      clear_screen()
      print("=====================================================")
      print("               REGISTERED USER GALLERY               ")
      print("=====================================================")
      if not gallery:
        print("No users currently registered in gallery_v2.pkl.")
      else:
        print(f"{'Index':<6} | {'User ID':<20} | {'Registration Timestamp'}")
        print("-" * 55)
        for idx, (uid, data) in enumerate(gallery.items(), 1):
          ts = data.get("enrolled_at", "N/A")
          print(f"{idx:<6} | {uid:<20} | {ts}")
      print("=====================================================")
      input("\nPress Enter to return...")
    elif choice == "3":
      clear_screen()
      print("=====================================================")
      print("                 DELETE USER PROFILE                 ")
      print("=====================================================")
      uid_to_del = input("Enter User ID to remove: ").strip()
      if uid_to_del in gallery:
        del gallery[uid_to_del]
        save_gallery(gallery)
        print(f"[DELETED] User '{uid_to_del}' removed from gallery.")
      else:
        print(f"[ERROR] User ID '{uid_to_del}' not found in gallery.")
      input("\nPress Enter to return...")
    elif choice == "0":
      break


# -----------------------------------------------------------------------------
# [3] ATTENDANCE LOGS & REPORTS
# -----------------------------------------------------------------------------
def attendance_reports():
  config = load_config()
  db_path = config["DB_PATH"]
  csv_dir = config["CSV_DIR"]

  while True:
    clear_screen()
    print("=====================================================")
    print("              ATTENDANCE LOGS & REPORTS              ")
    print("=====================================================")
    print("[1]  View Today's Attendance (SQLite)")
    print("[2]  Search Attendance by User ID")
    print("[3]  Search Attendance by Date Range")
    print("[4]  Open Google Drive Logs Path")
    print("[0]  Back to Main Menu")
    print("=====================================================")

    choice = input("Select an option [0-4]: ").strip()

    if choice == "1":
      clear_screen()
      today_str = datetime.now().strftime("%Y-%m-%d")
      print(f"=== TODAY'S LOGS ({today_str}) ===")
      if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            "SELECT date, time, user_id, score_full, score_perio, fused_score,"
            " status FROM attendance WHERE date = ?",
            (today_str,),
        )
        rows = c.fetchall()
        conn.close()

        if rows:
          print(
              f"{'Time':<10} | {'User ID':<15} | {'Full':<6} | {'Perio':<6} |"
              f" {'Fused':<6} | {'Status'}"
          )
          print("-" * 65)
          for r in rows:
            print(
                f"{r[1]:<10} | {r[2]:<15} | {r[3]:<6} | {r[4]:<6} | {r[5]:<6} |"
                f" {r[6]}"
            )
        else:
          print("No records logged today.")
      else:
        print("Database 'attendance.db' not found.")
      input("\nPress Enter to return...")

    elif choice == "2":
      clear_screen()
      search_uid = input("Enter User ID to search: ").strip()
      if search_uid and os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            "SELECT date, time, user_id, score_full, score_perio, fused_score,"
            " status FROM attendance WHERE user_id LIKE ? ORDER BY id DESC"
            " LIMIT 50",
            (f"%{search_uid}%",),
        )
        rows = c.fetchall()
        conn.close()

        if rows:
          print(
              f"{'Date':<11} | {'Time':<10} | {'User ID':<15} | {'Fused':<6} |"
              f" {'Status'}"
          )
          print("-" * 60)
          for r in rows:
            print(f"{r[0]:<11} | {r[1]:<10} | {r[2]:<15} | {r[5]:<6} | {r[6]}")
        else:
          print(f"No records found for '{search_uid}'.")
      input("\nPress Enter to return...")

    elif choice == "3":
      clear_screen()
      start_date = input("Enter Start Date (YYYY-MM-DD): ").strip()
      end_date = input("Enter End Date (YYYY-MM-DD): ").strip()
      if start_date and end_date and os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            "SELECT date, time, user_id, fused_score, status FROM attendance"
            " WHERE date BETWEEN ? AND ? ORDER BY id DESC",
            (start_date, end_date),
        )
        rows = c.fetchall()
        conn.close()

        if rows:
          print(
              f"{'Date':<11} | {'Time':<10} | {'User ID':<15} | {'Fused':<6} |"
              f" {'Status'}"
          )
          print("-" * 60)
          for r in rows:
            print(f"{r[0]:<11} | {r[1]:<10} | {r[2]:<15} | {r[3]:<6} | {r[4]}")
        else:
          print(f"No records found between {start_date} and {end_date}.")
      input("\nPress Enter to return...")

    elif choice == "4":
      if os.path.exists(csv_dir):
        os.startfile(csv_dir)
        print(f"[OPENED] Folder opened: {csv_dir}")
      else:
        print(f"[ERROR] Target path does not exist: {csv_dir}")
      input("\nPress Enter to return...")

    elif choice == "0":
      break


# -----------------------------------------------------------------------------
# [4] SYSTEM DIAGNOSTICS & CALIBRATION
# -----------------------------------------------------------------------------
def run_camera_diagnostics():
  config = load_config()
  quality_gate = QualityGate(
      blur_threshold=config["BLUR_THRESHOLD"],
      max_roll_deg=config["MAX_ROLL_DEG"],
  )
  mp_face = mp.solutions.face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.35
  )
  cap = cv2.VideoCapture(0)

  print("\n[DIAGNOSTICS] Camera & Quality Gate Inspection Active.")
  print("[INFO] Press 'q' to close window and return.")

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame_h, frame_w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
      for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x_px, y_px = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h)
        w_px, h_px = int(bbox.width * frame_w), int(bbox.height * frame_h)
        box = (max(0, x_px), max(0, y_px), max(1, w_px), max(1, h_px))

        kp = detection.location_data.relative_keypoints
        r_eye = (int(kp[0].x * frame_w), int(kp[0].y * frame_h))
        l_eye = (int(kp[1].x * frame_w), int(kp[1].y * frame_h))

        is_valid, msg, _ = quality_gate.inspect_and_align(
            frame, box, [r_eye, l_eye]
        )

        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cv2.rectangle(
            frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2
        )
        cv2.circle(frame, r_eye, 4, (255, 0, 0), -1)
        cv2.circle(frame, l_eye, 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            msg,
            (box[0], max(20, box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    cv2.imshow("Diagnostics - Quality Gate Inspection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


def run_pad_diagnostics():
  config = load_config()
  pad_detector = PresentationAttackDetector(
      fft_grid_threshold=config["FFT_GRID_THRESHOLD"]
  )
  mp_face = mp.solutions.face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.35
  )
  cap = cv2.VideoCapture(0)

  print("\n[DIAGNOSTICS] Anti-Spoofing (PAD) Live Inspection Active.")
  print("[INFO] Press 'q' to close window and return.")

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame_h, frame_w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
      for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x_px, y_px = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h)
        w_px, h_px = int(bbox.width * frame_w), int(bbox.height * frame_h)
        rx1, ry1 = max(0, x_px), max(0, y_px)
        rx2, ry2 = min(frame_w, x_px + w_px), min(frame_h, y_px + h_px)

        raw_crop = frame[ry1:ry2, rx1:rx2]
        if raw_crop.size > 0:
          is_spoof, reason = pad_detector.inspect_face_roi(raw_crop)
          color = (0, 0, 255) if is_spoof else (0, 255, 0)
          lbl = f"SPOOF ({reason})" if is_spoof else "LIVE REAL FACE"

          cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
          cv2.putText(
              frame,
              lbl,
              (rx1, max(20, ry1 - 10)),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.55,
              color,
              2,
          )

    cv2.imshow("Diagnostics - Anti-Spoofing (PAD)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


def diagnostics_menu():
  while True:
    clear_screen()
    print("=====================================================")
    print("          SYSTEM DIAGNOSTICS & CALIBRATION           ")
    print("=====================================================")
    print("[1]  Test Camera & Keypoint Alignment (Quality Gate)")
    print("[2]  Test Anti-Spoofing Detector (PAD FFT Lattice)")
    print("[0]  Back to Main Menu")
    print("=====================================================")

    choice = input("Select an option [0-2]: ").strip()

    if choice == "1":
      run_camera_diagnostics()
    elif choice == "2":
      run_pad_diagnostics()
    elif choice == "0":
      break


# -----------------------------------------------------------------------------
# [5] SETTINGS & THRESHOLD CONFIGURATION
# -----------------------------------------------------------------------------
def settings_menu():
  while True:
    clear_screen()
    config = load_config()
    print("=====================================================")
    print("        SETTINGS & THRESHOLD CONFIGURATION           ")
    print("=====================================================")
    print(f"[1]  Match Threshold      : {config['MATCH_THRESHOLD']}")
    print(f"[2]  Max Distance (Meters): {config['MAX_DISTANCE_METERS']}m")
    print(f"[3]  Consensus Frames (K) : {config['CONSENSUS_K']}")
    print(f"[4]  Cooldown Period      : {config['COOLDOWN_SEC']}s")
    print(f"[5]  Blur Threshold       : {config['BLUR_THRESHOLD']}")
    print(f"[6]  Max Head Roll Angle  : {config['MAX_ROLL_DEG']}°")
    print(f"[7]  FFT Anti-Spoof Limit : {config['FFT_GRID_THRESHOLD']}")
    print(f"[8]  Reset to Default Values")
    print("[0]  Back to Main Menu")
    print("=====================================================")

    choice = input("Select setting to modify [0-8]: ").strip()

    if choice == "1":
      val = input(
          f"Enter new Match Threshold (Current {config['MATCH_THRESHOLD']}): "
      ).strip()
      if val:
        config["MATCH_THRESHOLD"] = float(val)
    elif choice == "2":
      val = input(
          "Enter Max Distance in meters (Current"
          f" {config['MAX_DISTANCE_METERS']}): "
      ).strip()
      if val:
        config["MAX_DISTANCE_METERS"] = float(val)
    elif choice == "3":
      val = input(
          f"Enter Consensus K frames (Current {config['CONSENSUS_K']}): "
      ).strip()
      if val:
        config["CONSENSUS_K"] = int(val)
    elif choice == "4":
      val = input(
          f"Enter Cooldown in seconds (Current {config['COOLDOWN_SEC']}): "
      ).strip()
      if val:
        config["COOLDOWN_SEC"] = float(val)
    elif choice == "5":
      val = input(
          f"Enter Blur Threshold (Current {config['BLUR_THRESHOLD']}): "
      ).strip()
      if val:
        config["BLUR_THRESHOLD"] = float(val)
    elif choice == "6":
      val = input(
          f"Enter Max Roll Angle (Current {config['MAX_ROLL_DEG']}): "
      ).strip()
      if val:
        config["MAX_ROLL_DEG"] = float(val)
    elif choice == "7":
      val = input(
          f"Enter FFT Anti-Spoof Threshold (Current"
          f" {config['FFT_GRID_THRESHOLD']}): "
      ).strip()
      if val:
        config["FFT_GRID_THRESHOLD"] = float(val)
    elif choice == "8":
      config = DEFAULT_CONFIG.copy()
      print("[SETTINGS] Reset to defaults.")
    elif choice == "0":
      break

    save_config(config)


# -----------------------------------------------------------------------------
# [6] DATABASE & CLOUD SYNC MAINTENANCE
# -----------------------------------------------------------------------------
def maintenance_menu():
  while True:
    clear_screen()
    config = load_config()
    db_path = config["DB_PATH"]
    csv_dir = config["CSV_DIR"]

    print("=====================================================")
    print("        DATABASE & CLOUD SYNC MAINTENANCE            ")
    print("=====================================================")
    print(f"Database Path : {os.path.abspath(db_path)}")
    print(f"Google Drive  : {csv_dir}\n")
    print("[1]  Test Google Drive Connection & Path")
    print("[2]  Check Database Record Statistics")
    print("[3]  Backup SQLite Database")
    print("[4]  Clean Up Local Verification Snapshots")
    print("[0]  Back to Main Menu")
    print("=====================================================")

    choice = input("Select maintenance task [0-4]: ").strip()

    if choice == "1":
      clear_screen()
      print("Testing Google Drive target folder...")
      if os.path.exists(csv_dir):
        print(f"[ONLINE] Google Drive folder accessible: {csv_dir}")
        test_file = os.path.join(csv_dir, ".ping_test")
        try:
          with open(test_file, "w") as f:
            f.write("test")
          os.remove(test_file)
          print("[WRITE SUCCESS] Write permissions confirmed.")
        except Exception as e:
          print(f"[WRITE ERROR] Check permissions or sync locks: {e}")
      else:
        print(f"[OFFLINE] Drive folder not found at: {csv_dir}")
      input("\nPress Enter to return...")

    elif choice == "2":
      clear_screen()
      if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM attendance")
        count = c.fetchone()[0]
        size_kb = os.path.getsize(db_path) / 1024.0
        conn.close()

        print("=== SQLITE DATABASE STATS ===")
        print(f"Total Attendance Logs : {count} entries")
        print(f"Database File Size    : {size_kb:.2f} KB")
      else:
        print("Database 'attendance.db' not found.")
      input("\nPress Enter to return...")

    elif choice == "3":
      clear_screen()
      if os.path.exists(db_path):
        backup_filename = (
            f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        )
        shutil.copy(db_path, backup_filename)
        print(f"[BACKUP CREATED] Saved copy as: {backup_filename}")
      else:
        print("Database 'attendance.db' not found.")
      input("\nPress Enter to return...")

    elif choice == "4":
      clear_screen()
      snap_dir = "attendance_snapshots"
      if os.path.exists(snap_dir):
        files = os.listdir(snap_dir)
        confirm = (
            input(f"Delete {len(files)} snapshots in '{snap_dir}'? (y/n): ")
            .strip()
            .lower()
        )
        if confirm == "y":
          for f in files:
            os.remove(os.path.join(snap_dir, f))
          print(f"[CLEANED] Removed {len(files)} snapshot files.")
      else:
        print("No snapshot directory found.")
      input("\nPress Enter to return...")

    elif choice == "0":
      break


# -----------------------------------------------------------------------------
# MAIN CONTROL LOOP
# -----------------------------------------------------------------------------
def menu_loop():
  while True:
    display_menu()
    choice = input("Select an option [0-6]: ").strip()

    if choice == "1":
      launch_live_engine()
    elif choice == "2":
      gallery_management()
    elif choice == "3":
      attendance_reports()
    elif choice == "4":
      diagnostics_menu()
    elif choice == "5":
      settings_menu()
    elif choice == "6":
      maintenance_menu()
    elif choice == "0":
      print("\nShutting down Ocular AI TAS System. Goodbye!\n")
      sys.exit(0)
    else:
      input("\nInvalid option! Press Enter to try again...")


if __name__ == "__main__":
  menu_loop()