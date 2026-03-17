<<<<<<< HEAD
import cv2
import argparse
import uuid
import datetime
import sqlite3
import time
=======
"""
Real-time Tube Detection with Tracking
======================================
- Prefers ONNX model (then NCNN, TFLite, PyTorch)
- 5-second cooldown prevents duplicate counts
- Unique IDs: TUBE_001, TUBE_002, ...
- Only tube classes, confidence > 0.7
- ByteTrack tracking (same tube = one count)
- Optimized for Raspberry Pi
"""

import cv2
import argparse
import sqlite3
import time
import os
>>>>>>> 1a5e2cf (Initial commit or description of changes)
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from ultralytics import YOLO


def _normalize_name(s: str) -> str:
<<<<<<< HEAD
    """Lowercase and strip spaces/specials for stable name comparison."""
    return (
        s.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
    )


# ─────────────────────────────────────────────────────
# Threaded Video Capture (important for Raspberry Pi)
# ─────────────────────────────────────────────────────
class VideoCaptureAsync:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.running = True

=======
    return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "").replace("(", "").replace(")", "")


# ─────────────────────────────────────────────────────────────────────────────
# Threaded Video Capture (Raspberry Pi)
# ─────────────────────────────────────────────────────────────────────────────
class VideoCaptureAsync:
    def __init__(self, source):
        if os.name == "nt" and isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.running = True
>>>>>>> 1a5e2cf (Initial commit or description of changes)
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.grabbed, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


<<<<<<< HEAD
# ─────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────
def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            track_id INTEGER,
            brand_name TEXT,
            confidence REAL
        )
    ''')

    conn.commit()
    return conn


def log_inspection(conn, track_id, class_name, conf):
    check_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO inspections (id, timestamp, track_id, brand_name, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (check_id, timestamp, track_id, class_name, round(conf, 2)))

    conn.commit()
    return check_id


# ─────────────────────────────────────────────────────
# Load valid tube classes
# ─────────────────────────────────────────────────────
def load_valid_classes(data_dir: Path):

    classes_file = data_dir / "classes.txt"

    if not classes_file.exists():
        return []

    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines() if line.strip()]


# ─────────────────────────────────────────────────────
# Main System
# ─────────────────────────────────────────────────────
def run_system(source_input,
               imgsz=320,
               min_conf=0.9,
               frame_skip=2,
               min_box_area=4000,
               nms_threshold=0.7,
               dedupe=False,
               debug=False,
               headless=False):

    base_dir = Path(__file__).resolve().parent.parent

    # Use the latest trained weights from brand_experiment3
    onnx_model_path = base_dir / "runs/detect/brand_experiment3/weights/best.onnx"
    ncnn_model_path = base_dir / "runs/detect/brand_experiment3/weights/best_ncnn_model"
    tflite_model_path = base_dir / "runs/detect/brand_experiment3/weights/best_saved_model/best_float32.tflite"
    pt_model_path = base_dir / "runs/detect/brand_experiment3/weights/best.pt"

    # Clamp to a high-confidence operating point
    if min_conf < 0.9:
        print(f"🔧 Bumping min_conf from {min_conf:.2f} to 0.90 for high-confidence tube-only detections.")
        min_conf = 0.9

    # ───── Load model ─────
    if onnx_model_path.exists():
        print("🚀 Loading ONNX model")
        model = YOLO(str(onnx_model_path), task='detect')

    elif ncnn_model_path.exists():
        print("🚀 Loading NCNN model")
        model = YOLO(str(ncnn_model_path), task='detect')

    elif tflite_model_path.exists():
        print("🚀 Loading TFLite model")
        model = YOLO(str(tflite_model_path), task='detect')

    elif pt_model_path.exists():
        print("✅ Loading PyTorch model")
        model = YOLO(str(pt_model_path))

    else:
        raise RuntimeError(
            "No trained tube-detection model found in runs/detect/brand_experiment3/weights.\n"
            "Please place your latest best.pt / ONNX / NCNN / TFLite weights there and retry."
        )

    # ───── Load valid classes ─────
    valid_classes = load_valid_classes(base_dir / "data" / "dataset(tubes)")
    normalized_valid = {_normalize_name(c) for c in valid_classes}

    # Map model label variants to canonical class names from classes.txt
    alias_map = {
        # Model label '(Valbet)' → classes.txt 'valbet'
        "valbet": "valbet",
        # Model label 'S K Kant' → classes.txt 'silverkant'
        "skkant": "silverkant",
    }

    db_path = base_dir / "data/inspections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Text log file for every detected tube
    log_file = base_dir / "data" / "detections.log"

    conn = setup_db(db_path)

    # ───── Camera setup ─────
    if source_input.isnumeric():
        source_input = int(source_input)
        cap = VideoCaptureAsync(source_input)
        print("📹 Using threaded capture")

    else:
        cap = cv2.VideoCapture(source_input)

    logged_objects = set()
    total_tubes_logged = 0  # running count: +1 each time a tube is logged to DB

    frame_count = 0

=======
# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────
def setup_db(conn: sqlite3.Connection):
    """Create tube_detections table: tube_id, timestamp, brand_name, confidence, track_id."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tube_detections (
            tube_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            brand_name TEXT,
            confidence REAL,
            track_id INTEGER
        )
    """)
    conn.commit()


def get_next_tube_id(conn: sqlite3.Connection) -> str:
    """TUBE_001, TUBE_002, ..."""
    cursor = conn.cursor()
    cursor.execute("SELECT tube_id FROM tube_detections ORDER BY tube_id DESC LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        return "TUBE_001"
    try:
        num = int(str(row[0]).replace("TUBE_", ""))
        return f"TUBE_{num + 1:03d}"
    except (ValueError, AttributeError):
        return "TUBE_001"


def log_tube(conn: sqlite3.Connection, tube_id: str, brand_name: str, confidence: float, track_id: int):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO tube_detections (tube_id, timestamp, brand_name, confidence, track_id)
        VALUES (?, ?, ?, ?, ?)
    """, (tube_id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), brand_name, round(confidence, 2), track_id))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Load valid tube classes (only tubes, ignore person/background/noise)
# ─────────────────────────────────────────────────────────────────────────────
def load_valid_classes(base_dir: Path) -> set:
    classes_file = base_dir / "data" / "dataset(tubes)" / "classes.txt"
    if not classes_file.exists():
        return {_normalize_name(c) for c in ["valbet", "araldite", "beutiful-n", "dk gel", "silverkant", "cani-maks"]}
    with open(classes_file, "r", encoding="utf-8") as f:
        return {_normalize_name(line.strip()) for line in f if line.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# Main System
# ─────────────────────────────────────────────────────────────────────────────
def run_system(
    source_input,
    imgsz=320,
    min_conf=0.7,
    frame_skip=2,
    min_box_area=3000,
    cooldown_sec=5.0,
    headless=False,
    run_name="balanced_run",
):
    base_dir = Path(__file__).resolve().parent.parent
    weights_dir = base_dir / "runs" / "detect" / run_name / "weights"

    # Prefer ONNX (faster on Pi), then NCNN, TFLite, PyTorch
    onnx_path = weights_dir / "best.onnx"
    ncnn_path = weights_dir / "best_ncnn_model"
    tflite_path = weights_dir / "best_saved_model" / "best_float32.tflite"
    pt_path = weights_dir / "best.pt"

    if onnx_path.exists():
        print("🚀 Loading ONNX model")
        model = YOLO(str(onnx_path), task="detect")
    elif ncnn_path.exists():
        print("🚀 Loading NCNN model")
        model = YOLO(str(ncnn_path), task="detect")
    elif tflite_path.exists():
        print("🚀 Loading TFLite model")
        model = YOLO(str(tflite_path), task="detect")
    elif pt_path.exists():
        print("✅ Loading PyTorch model")
        model = YOLO(str(pt_path))
    else:
        raise RuntimeError(
            f"No model in runs/detect/{run_name}/weights. "
            f"Export ONNX: model.export(format='onnx') or train with --name {run_name}"
        )

    normalized_valid = load_valid_classes(base_dir)
    alias_map = {
        "valbet": "valbet", "val-bet": "valbet", "val bet": "valbet",
        "silverkant": "silverkant", "silver-kant": "silverkant", "skkant": "silverkant",
        "dk_gel": "dk gel", "dk gel": "dk gel",
        "cani-maks": "cani-maks", "cani maks": "cani-maks",
        "beutiful-n": "beutiful-n", "beautiful-n": "beutiful-n",
        "araldite": "araldite", "halobet": "halobet",
    }

    db_path = base_dir / "data" / "tube_detections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    setup_db(conn)

    # Cooldown & tracking
    last_detection_time = 0.0
    logged_track_ids = set()
    tube_count = 0

    source_int = int(source_input) if str(source_input).isdigit() else source_input
    cap = VideoCaptureAsync(source_int) if isinstance(source_int, int) else cv2.VideoCapture(source_input)

    if not cap.isOpened():
        print("Camera not opened. Try --source 0 or 1")
        conn.close()
        return

    print("=" * 50)
    print(f"Tube Detection | Model: {run_name} | Cooldown: {cooldown_sec}s")
    print("Press Q to quit")
    print("=" * 50)

    frame_count = 0
>>>>>>> 1a5e2cf (Initial commit or description of changes)
    fps_start = time.time()
    fps_counter = 0
    display_fps = 0.0

<<<<<<< HEAD
    print("🎥 Camera Active. Press 'Q' to quit.")

    if not headless:
        cv2.namedWindow("ML Brand Detector", cv2.WINDOW_NORMAL)

    # ─────────────────────────────
    # Main loop
    # ─────────────────────────────
    while cap.isOpened():

        success, frame = cap.read()

=======
    if not headless:
        cv2.namedWindow("ML Brand Detector", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
>>>>>>> 1a5e2cf (Initial commit or description of changes)
        if not success or frame is None:
            break

        frame_count += 1
<<<<<<< HEAD

        if frame_count % frame_skip != 0:
            continue

        # ───── YOLO Tracking (IMPORTANT FIX) ─────
        with torch.no_grad():

=======
        if frame_count % frame_skip != 0:
            if not headless:
                cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("ML Brand Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        with torch.no_grad():
>>>>>>> 1a5e2cf (Initial commit or description of changes)
            results = model.track(
                frame,
                conf=min_conf,
                imgsz=imgsz,
                persist=True,
                tracker="bytetrack.yaml",
<<<<<<< HEAD
                verbose=False
            )

        detections = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
            else:
                track_ids = np.full(len(boxes), -1, dtype=int)

            for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):

                x1, y1, x2, y2 = box

                raw_name = model.names[int(cls)]
                norm_model_name = _normalize_name(raw_name)
                # Resolve to canonical label if we know an alias
                canonical = alias_map.get(norm_model_name, raw_name.lower())
                if _normalize_name(canonical) not in normalized_valid:
                    continue

                area = (x2 - x1) * (y2 - y1)

                if area < min_box_area:
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                detections.append((canonical, conf, (x1, y1, x2, y2), track_id))

                if not headless:

                    # Yellow box around detected tube
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    label = f"ID:{track_id} {canonical}"

                    # Yellow label text
                    cv2.putText(frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2)

        # ───── Logging ─────
        if detections:

            names = [d[0] for d in detections]
            track_ids = [d[3] for d in detections]

            unique_ids = {tid for tid in track_ids if tid != -1}

            for brand_name, conf, _, track_id in detections:

                if conf < min_conf:
                    continue

                if dedupe:
                    key = (track_id, brand_name)

                    if key in logged_objects:
                        continue

                    logged_objects.add(key)

                uuid_code = log_inspection(conn, track_id, brand_name, conf)
                total_tubes_logged += 1

                # Append to plain-text log so every detection is recorded
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    with open(log_file, "a", encoding="utf-8") as lf:
                        lf.write(f"{timestamp},track={track_id},tube={brand_name},conf={conf:.2f},uuid={uuid_code}\n")
                except Exception:
                    # Avoid breaking the main loop if logging fails
                    pass

                print(f"Logged to DB and file: {brand_name} | Confidence: {conf:.2f}")
            print(f"Detected Tube: {', '.join(names)} | Total tubes: {total_tubes_logged}")

        # ───── FPS counter ─────
        fps_counter += 1

        elapsed = time.time() - fps_start

        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(frame,
                    f"FPS: {display_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        if not headless:

            cv2.imshow("ML Brand Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:

            if frame_count % 20 == 0:
                print(f"⚡ FPS: {display_fps:.1f}")

    cap.release()

    if not headless:
        cv2.destroyAllWindows()

    conn.close()

    print(f"\n✅ Session complete. Final FPS: {display_fps:.1f}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--min-conf", type=float, default=0.85)
    parser.add_argument("--min-area", type=int, default=4000)
    parser.add_argument("--nms-thresh", type=float, default=0.7)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()

    run_system(
        args.source,
=======
                verbose=False,
            )

        best_detection = None
        best_conf = 0.0

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = (
                results[0].boxes.id.int().cpu().numpy()
                if results[0].boxes.id is not None
                else np.full(len(boxes), -1, dtype=int)
            )

            for box, cls_id, conf, track_id in zip(boxes, clss, confs, track_ids):
                if conf < min_conf:
                    continue
                raw_name = model.names.get(int(cls_id), "")
                norm = _normalize_name(raw_name)
                canonical = alias_map.get(norm, raw_name.lower())
                if _normalize_name(canonical) not in normalized_valid:
                    continue
                x1, y1, x2, y2 = box
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue
                if track_id >= 0 and track_id in logged_track_ids:
                    continue
                if conf > best_conf:
                    best_conf = float(conf)
                    best_detection = (canonical, float(conf), (int(x1), int(y1), int(x2), int(y2)), int(track_id) if track_id >= 0 else -1)

        now = time.time()
        cooldown_elapsed = (now - last_detection_time) >= cooldown_sec

        if best_detection and cooldown_elapsed:
            brand_name, conf, (x1, y1, x2, y2), track_id = best_detection
            tube_id = get_next_tube_id(conn)
            log_tube(conn, tube_id, brand_name, conf, track_id)
            if track_id >= 0:
                logged_track_ids.add(track_id)
            last_detection_time = now
            tube_count += 1
            print(f"[{tube_id}] {brand_name} | conf={conf:.2f} | total={tube_count}")
            if not headless:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{tube_id} {brand_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif best_detection and not headless:
            _, _, (x1, y1, x2, y2), _ = best_detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, "cooldown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if len(logged_track_ids) > 20:
            logged_track_ids.clear()

        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            display_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tubes: {tube_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if cooldown_elapsed:
            cv2.putText(frame, "READY", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            rem = cooldown_sec - (now - last_detection_time)
            cv2.putText(frame, f"Cooldown: {rem:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if not headless:
            cv2.imshow("ML Brand Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    conn.close()
    print(f"\nDone. Total tubes: {tube_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="1", help="Camera index")
    parser.add_argument("--run", type=str, default="balanced_run", help="Model run (runs/detect/<run>/weights)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between detections")
    parser.add_argument("--min-conf", type=float, default=0.70, help="Min confidence")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size (320=fast)")
    parser.add_argument("--min-area", type=int, default=3000, help="Min box area")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    run_system(
        source_input=args.source,
>>>>>>> 1a5e2cf (Initial commit or description of changes)
        imgsz=args.imgsz,
        min_conf=args.min_conf,
        frame_skip=args.skip,
        min_box_area=args.min_area,
<<<<<<< HEAD
        nms_threshold=args.nms_thresh,
        dedupe=args.dedupe,
        debug=args.debug,
        headless=args.headless,
    )
=======
        cooldown_sec=args.cooldown,
        headless=args.headless,
        run_name=args.run,
    )
>>>>>>> 1a5e2cf (Initial commit or description of changes)
