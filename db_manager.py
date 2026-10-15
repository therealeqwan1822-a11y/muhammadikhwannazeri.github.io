# db_manager.py
import csv
import os
import sqlite3
import time
from datetime import datetime


class DatabaseManager:

    def __init__(
        self, db_path="attendance.db", csv_dir=r"G:\My Drive\Attendance_Logs"
    ):
        self.db_path = db_path
        self.csv_dir = os.path.abspath(csv_dir)

        # Create Google Drive target folder if it doesn't exist
        try:
            os.makedirs(self.csv_dir, exist_ok=True)
            print(f"[CLOUD LOG] Active Google Drive target: {self.csv_dir}")
        except Exception as e:
            print(f"[CLOUD WARNING] Check Google Drive path: {e}")

        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                user_id TEXT,
                score_full REAL,
                score_perio REAL,
                fused_score REAL,
                status TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_attendance(
        self, user_id, score_full, score_perio, fused_score, status
    ):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # 1. Local SQLite Backup Log
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO attendance (date, time, user_id, score_full, score_perio, fused_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date_str,
                    time_str,
                    user_id,
                    score_full,
                    score_perio,
                    fused_score,
                    status,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB ERROR] SQLite write error: {e}")

        # 2. Google Drive Cloud CSV Logging with Retry Loop
        csv_path = os.path.join(self.csv_dir, f"attendance_{date_str}.csv")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                file_exists = os.path.exists(csv_path)

                with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "Date",
                            "Time",
                            "User_ID",
                            "Score_Full",
                            "Score_Perio",
                            "Fused_Score",
                            "Status",
                        ])
                    writer.writerow([
                        date_str,
                        time_str,
                        user_id,
                        score_full,
                        score_perio,
                        fused_score,
                        status,
                    ])

                    # Force OS buffer to flush directly to Google Drive stream
                    f.flush()
                    os.fsync(f.fileno())

                print(
                    f"[CLOUD LOG SUCCESS] Recorded {user_id} at {time_str} ->"
                    f" Google Drive"
                )
                break

            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.4)  # Wait for Google Drive sync lock to free
                else:
                    print(
                        f"\n[CLOUD LOCK ERROR] '{csv_path}' is open in Excel or"
                        " locked by Google Drive Sync!"
                    )
                    print(
                        "[ACTION REQUIRED] Please close Excel to allow cloud"
                        " sync.\n"
                    )
            except Exception as e:
                print(f"[CLOUD CSV ERROR] {e}")
                break