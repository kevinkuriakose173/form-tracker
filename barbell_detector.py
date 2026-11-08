"""Inference adapter for a custom, single-keypoint bar-end model."""
from pathlib import Path
import os

import numpy as np


class BarbellDetector:
    def __init__(self, weights, confidence=0.25, keypoint_confidence=0.5, device=None):
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.is_file():
            raise ValueError(f"Custom weights not found: {weights}")
        self.model = YOLO(str(weights))
        shape = self.model.model.yaml.get("kpt_shape")
        if self.model.task != "pose" or list(shape or []) != [1, 3]:
            raise ValueError("Expected custom pose weights with kpt_shape [1, 3], not a human-pose model")
        if dict(self.model.names) != {0: "bar_end"}:
            raise ValueError("Expected exactly one class named bar_end")
        if not 0 <= confidence <= 1 or not 0 <= keypoint_confidence <= 1:
            raise ValueError("Confidence thresholds must be between 0 and 1")
        self.confidence = confidence
        self.keypoint_confidence = keypoint_confidence
        self.device = device

    def locate(self, frame, wrist=None):
        """Return (accepted x/y or None, score). No template or wrist fallback."""
        options = {"verbose": False, "conf": self.confidence, "classes": [0]}
        if self.device is not None:
            options["device"] = self.device
        result = self.model.predict(frame, **options)[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return None, None
        box_scores = result.boxes.conf.cpu().numpy()
        points = result.keypoints.xy.cpu().numpy()
        point_scores = result.keypoints.conf
        if point_scores is None:
            return None, None
        point_scores = point_scores.cpu().numpy()
        h, w = frame.shape[:2]
        for index in np.argsort(-box_scores):
            x, y = map(float, points[index, 0])
            score = float(point_scores[index, 0])
            if (np.isfinite([x, y, score]).all() and score >= self.keypoint_confidence
                    and 0 <= x < w and 0 <= y < h):
                return (x, y), score
        return None, None
