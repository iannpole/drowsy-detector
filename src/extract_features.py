"""Extract EAR/MAR/head-pose features from image datasets into pipeline CSVs."""

import argparse
import csv
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
COLUMNS = [
    "subject",
    "split",
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
# Calibrated on NTHU open-eye frames: EAR 0.35 = open, 0.18 = closed.
BASELINE_EAR = 0.35
CLOSED_RATIO = 0.55
BATCH = 5000


def parse_nthu(path):
    """002_glasses_sleepyCombination_2049_drowsy.jpg -> (002, label)."""
    match = re.match(r"^(\d+)_.*_(drowsy|notdrowsy)\.jpe?g$", path.name, re.IGNORECASE)
    if not match:
        return None
    subject = f"nthu{match.group(1)}"
    label = "drowsy" if match.group(2).lower() == "drowsy" else "alert"
    return subject, label


def parse_uta(path):
    """train/fatigue/image_0162.jpg -> (uta, drowsy, train)."""
    folder = path.parent.name.lower()
    label = {"active": "alert", "fatigue": "drowsy"}.get(folder)
    if label is None:
        return None
    return "uta", label, path.parent.parent.name.lower()


def iter_images(root, parser):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parsed = parser(path)
        if parsed:
            yield path, parsed[0], parsed[1], parsed[2] if len(parsed) > 2 else ""


def extract(landmarker, path, subject, label, split):
    frame = cv2.imread(str(path))
    if frame is None:
        return None
    height, width = frame.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(image)
    if not result.face_landmarks:
        return None
    marks = result.face_landmarks[0]
    ear_value = (features.ear(marks, features.LEFT_EYE, width, height) + features.ear(marks, features.RIGHT_EYE, width, height)) / 2.0
    pitch, yaw, _ = features.head_pose(marks, width, height)
    return {
        "subject": subject,
        "split": split,
        "ear_mean": ear_value,
        "ear_std": 0.0,
        "ear_norm": ear_value / BASELINE_EAR,
        "mar_mean": features.mar(marks, width, height),
        "perclos": 1.0 if ear_value < BASELINE_EAR * CLOSED_RATIO else 0.0,
        "blink_rate": 0.0,
        "max_closed_sec": 0.0,
        "pitch_mean": pitch,
        "pitch_std": 0.0,
        "yaw_mean": yaw,
        "yaw_std": 0.0,
        "label": label,
    }


def run(dataset, root, out):
    parser = parse_nthu if dataset == "nthu" else parse_uta
    items = list(iter_images(root, parser))
    if not items:
        raise SystemExit(f"No labelled images found under {root}")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    written = missed = 0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker, out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for path, subject, label, split in items:
            row = extract(landmarker, path, subject, label, split)
            if row is None:
                missed += 1
                continue
            writer.writerow(row)
            written += 1
            if written % BATCH == 0:
                print(f"  {written}/{len(items)} kept, {missed} without face", flush=True)
    print(f"{dataset}: {written} rows -> {out} ({missed} images without a detectable face)")


def main():
    parser = argparse.ArgumentParser(description="Extract EAR/MAR/head pose from image datasets.")
    parser.add_argument("dataset", choices=["nthu", "uta"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not MODEL_PATH.is_file():
        raise SystemExit(f"MediaPipe model not found: {MODEL_PATH}")
    run(args.dataset, args.root, args.out)


if __name__ == "__main__":
    main()
