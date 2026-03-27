# =========================
# FINAL VERSION (UI + FIXED)
# =========================

import cv2
import argparse
import sqlite3
import time
import datetime
import numpy as np
import torch
from pathlib import Path
from threading import Thread
from ultralytics import YOLO


# =========================
# Video Thread
# =========================
class VideoCaptureAsync:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.grabbed, self.frame = self.cap.read()
        self.running = True
        Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.frame = frame

    def read(self):
        return True, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()


# =========================
# DB Setup
# =========================
def setup_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tube_detections (
            tube_id TEXT PRIMARY KEY,
            timestamp TEXT,
            brand_name TEXT,
            confidence REAL,
            track_id INTEGER
        )
    """)

    conn.commit()
    return conn


def get_next_id(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT tube_id FROM tube_detections ORDER BY tube_id DESC LIMIT 1")
    row = cursor.fetchone()

    if not row:
        return "TUBE_001"

    num = int(row[0].split("_")[1])
    return f"TUBE_{num+1:03d}"


def log(conn, tid, brand, conf, track_id):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO tube_detections VALUES (?, ?, ?, ?, ?)
    """, (
        tid,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        brand,
        round(conf, 2),
        track_id
    ))
    conn.commit()


# =========================
# MAIN SYSTEM
# =========================
def run():

    base_dir = Path(__file__).resolve().parent.parent

    # ✅ FIXED PATHS
    weights = base_dir / "runs/detect/balanced_run/weights/best.onnx"
    db_path = base_dir / "data/tube_detections.db"

    print("📦 DB:", db_path)
    print("🤖 Model:", weights)

    conn = setup_db(db_path)

    model = YOLO(str(weights), task="detect")

    cap = VideoCaptureAsync(0)

    cooldown = 5
    last_time = 0
    total = 0

    fps_count = 0
    fps_start = time.time()
    fps = 0

    print("🚀 Running... Press Q to quit")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            ids = results[0].boxes.id
            ids = ids.cpu().numpy() if ids is not None else [-1]*len(boxes)

            for box, cls, conf, tid in zip(boxes, clss, confs, ids):

                if conf < 0.7:
                    continue

                x1, y1, x2, y2 = map(int, box)
                brand = model.names[int(cls)]

                # Draw box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{brand} {conf:.2f}", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                now = time.time()

                if now - last_time >= cooldown:
                    tube_id = get_next_id(conn)

                    log(conn, tube_id, brand, conf, int(tid))

                    print(f"🔔 {tube_id} | {brand} | {conf:.2f}")

                    total += 1
                    last_time = now

        # FPS
        fps_count += 1
        if time.time() - fps_start >= 1:
            fps = fps_count / (time.time() - fps_start)
            fps_count = 0
            fps_start = time.time()

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Tubes: {total}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run()