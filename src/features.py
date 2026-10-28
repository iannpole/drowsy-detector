from collections import deque
from math import hypot

import cv2
import numpy as np

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def _point(landmark, w, h):
    return np.array((landmark.x * w, landmark.y * h), dtype=np.float64)


def _distance(a, b):
    return hypot(*(a - b))


def ear(landmarks, indices, w, h):
    points = [_point(landmarks[index], w, h) for index in indices]
    horizontal = 2.0 * _distance(points[0], points[3])
    if horizontal == 0:
        return 0.0
    return (_distance(points[1], points[5]) + _distance(points[2], points[4])) / horizontal


def mar(landmarks, w, h):
    upper = _point(landmarks[13], w, h)
    lower = _point(landmarks[14], w, h)
    left = _point(landmarks[78], w, h)
    right = _point(landmarks[308], w, h)
    width = _distance(left, right)
    return _distance(upper, lower) / width if width else 0.0


def head_pose(landmarks, w, h):
    image_points = np.array(
        [
            _point(landmarks[1], w, h),
            _point(landmarks[152], w, h),
            _point(landmarks[33], w, h),
            _point(landmarks[263], w, h),
            _point(landmarks[61], w, h),
            _point(landmarks[291], w, h),
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )
    focal_length = float(w)
    camera_matrix = np.array(
        [[focal_length, 0.0, w / 2.0], [0.0, focal_length, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


class WindowFeatures:
    def __init__(self, window_seconds=3.0, closed_threshold=0.2):
        self.window_seconds = float(window_seconds)
        self.closed_threshold = float(closed_threshold)
        self._samples = deque()

    def add(self, timestamp, ear_value, mar_value, pitch, yaw):
        sample = (
            float(timestamp),
            float(ear_value),
            float(mar_value),
            float(pitch),
            float(yaw),
        )
        self._samples.append(sample)
        cutoff = sample[0] - self.window_seconds
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def clear(self):
        self._samples.clear()

    def compute(self):
        if not self._samples:
            return {
                key: 0.0
                for key in (
                    "ear_mean",
                    "ear_std",
                    "mar_mean",
                    "perclos",
                    "blink_rate",
                    "max_closed_sec",
                    "pitch_mean",
                    "pitch_std",
                    "yaw_mean",
                    "yaw_std",
                )
            }

        values = np.asarray([sample[1:] for sample in self._samples], dtype=np.float64)
        timestamps = np.asarray([sample[0] for sample in self._samples], dtype=np.float64)
        closed = values[:, 0] < self.closed_threshold
        blink_count = int(np.count_nonzero((~closed[:-1]) & closed[1:])) if len(closed) > 1 else 0
        span = max(timestamps[-1] - timestamps[0], 1e-9)

        longest = 0.0
        start = None
        for timestamp, is_closed in zip(timestamps, closed):
            if is_closed and start is None:
                start = timestamp
            elif not is_closed and start is not None:
                longest = max(longest, timestamp - start)
                start = None
        if start is not None:
            longest = max(longest, timestamps[-1] - start)

        return {
            "ear_mean": float(values[:, 0].mean()),
            "ear_std": float(values[:, 0].std()),
            "mar_mean": float(values[:, 1].mean()),
            "perclos": float(closed.mean()),
            "blink_rate": float(blink_count * 60.0 / span),
            "max_closed_sec": float(longest),
            "pitch_mean": float(values[:, 2].mean()),
            "pitch_std": float(values[:, 2].std()),
            "yaw_mean": float(values[:, 3].mean()),
            "yaw_std": float(values[:, 3].std()),
        }
