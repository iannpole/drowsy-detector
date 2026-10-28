import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import LEFT_EYE, RIGHT_EYE, WindowFeatures, ear, head_pose, mar

COLUMNS = [
    "timestamp",
    "subject",
    "ear_mean",
    "ear_std",
    "ear_norm",
    "mar_mean",
    "perclos",
    "blink_rate",
    "max_closed_sec",
    "pitch_mean",
    "pitch_std",
    "yaw_mean",
    "yaw_std",
    "label",
]
LABELS = {ord("0"): "alert", ord("1"): "drowsy", ord("2"): "distracted"}
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect numeric drowsiness features.")
    parser.add_argument("--subject", required=True, help="Subject identifier")
    parser.add_argument("--out", type=Path, help="Output CSV (default: data/<subject>.csv)")
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic data")
    return parser.parse_args()


def output_path(args):
    path = args.out or Path("data") / f"{args.subject}.csv"
    return path if path.is_absolute() else ROOT / path


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_row(timestamp, subject, window, baseline, label):
    values = window.compute()
    row = {
        "timestamp": f"{timestamp:.3f}",
        "subject": subject,
        **{key: f"{value:.6f}" for key, value in values.items()},
        "ear_norm": f"{values['ear_mean'] / baseline:.6f}",
        "label": label,
    }
    return {column: row[column] for column in COLUMNS}


def run_dry(args, path):
    rng = np.random.default_rng(42)
    baseline_samples = 0.30 + rng.normal(0.0, 0.006, 200)
    baseline = float(baseline_samples.mean())
    window = WindowFeatures(closed_threshold=baseline * 0.7)
    start = time.time()
    rows = 0

    for index in range(60):
        elapsed = index * 0.1
        if elapsed < 2.0:
            label, base_ear, pitch, yaw = "alert", baseline, 1.0, 1.5
        elif elapsed < 4.0:
            label, base_ear, pitch, yaw = "drowsy", baseline * 0.55, 5.0, 2.0
        else:
            label, base_ear, pitch, yaw = "distracted", baseline * 0.95, 3.0, 25.0
        ear_value = base_ear + rng.normal(0.0, 0.008)
        mar_value = 0.08 + (0.12 if label == "drowsy" else 0.0) + rng.normal(0.0, 0.005)
        window.add(elapsed, ear_value, mar_value, pitch + rng.normal(0.0, 1.0), yaw + rng.normal(0.0, 1.5))
        if index % 5 == 0:
            append_row(path, make_row(start + elapsed, args.subject, window, baseline, label))
            rows += 1

    print(f"Dry run complete: {rows} rows saved to {path}")
    print(f"Synthetic EAR baseline: {baseline:.6f}")


def average_ear(landmarks, width, height):
    return (ear(landmarks, LEFT_EYE, width, height) + ear(landmarks, RIGHT_EYE, width, height)) / 2.0


def run_camera(args, path):
    model_path = ROOT / "models" / "face_landmarker.task"
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    window = WindowFeatures()
    calibration = []
    calibration_started = time.monotonic()
    last_saved = 0.0
    rows = 0
    label = "alert"
    paused = False
    baseline = None

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a webcam frame")
                now = time.monotonic()
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(now * 1000))
                current = None

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    ear_value = average_ear(landmarks, width, height)
                    mar_value = mar(landmarks, width, height)
                    pitch, yaw, _ = head_pose(landmarks, width, height)
                    current = ear_value, mar_value, pitch, yaw

                    if baseline is None:
                        calibration.append(ear_value)
                    elif not paused:
                        window.add(now, ear_value, mar_value, pitch, yaw)
                        if now - last_saved >= 0.5:
                            append_row(path, make_row(time.time(), args.subject, window, baseline, label))
                            rows += 1
                            last_saved = now

                if baseline is None and now - calibration_started >= 10.0:
                    if not calibration:
                        calibration_started = now
                    else:
                        baseline = float(np.median(calibration))
                        window.closed_threshold = baseline * 0.7
                        window.clear()
                        last_saved = now

                if baseline is None:
                    remaining = max(0.0, 10.0 - (now - calibration_started))
                    status = f"CALIBRATION: eyes open, look forward ({remaining:.1f}s)"
                else:
                    status = f"Label: {label}" + (" [PAUSED]" if paused else "")
                cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                if current:
                    cv2.putText(
                        frame,
                        f"EAR {current[0]:.3f}  MAR {current[1]:.3f}  Pitch {current[2]:.1f}  Yaw {current[3]:.1f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )
                cv2.putText(frame, f"Rows saved: {rows}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow("Drowsiness Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key in LABELS:
                    label = LABELS[key]
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Collection complete: {rows} rows saved to {path}")


def main():
    args = parse_args()
    path = output_path(args)
    try:
        if args.dry_run:
            run_dry(args, path)
        else:
            run_camera(args, path)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
