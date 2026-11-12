from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from board_detector import BoardDetector
from config import Config


_DEFAULT_TUNING_CANDIDATES = [
    {
        "contrast_threshold": 12.0,
        "mean_diff_threshold": 6.0,
        "std_dev_threshold": 10.0,
        "white_margin": 14.0,
        "occupied_ratio": 0.08,
    },
    {
        "contrast_threshold": 10.0,
        "mean_diff_threshold": 5.0,
        "std_dev_threshold": 8.0,
        "white_margin": 12.0,
        "occupied_ratio": 0.06,
    },
    {
        "contrast_threshold": 14.0,
        "mean_diff_threshold": 7.0,
        "std_dev_threshold": 12.0,
        "white_margin": 16.0,
        "occupied_ratio": 0.10,
    },
]


@dataclass
class Detection:
    """Single occupancy detection with rectangle and confidence data."""

    class_id: int
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    center: tuple[float, float]


class PieceDetector:
    """Detect occupied board squares and distinguish black/white pieces without a trained model."""

    def __init__(
        self,
        board_detector: BoardDetector | None = None,
        model_path: str | None = None,
        params: dict[str, float] | None = None,
    ) -> None:
        """Create the lightweight occupancy detector.

        The detector does not use any YOLO model anymore; it relies on the already
        calibrated chessboard geometry and a simple difference-from-board-background
        test for each square. The output is occupancy-only because we do not need to
        distinguish white from black in order to identify the board state.
        """
        self.board_detector = board_detector
        self.model_path: str = model_path or Config.MODEL_PATH
        self.model = None
        self.class_names = {0: "occupied"}
        self.params = params or {
            "contrast_threshold": 12.0,
            "mean_diff_threshold": 6.0,
            "std_dev_threshold": 10.0,
            "white_margin": 14.0,
            "occupied_ratio": 0.08,
        }

    @staticmethod
    def classify_roi(roi: np.ndarray, params: dict[str, float] | None = None) -> tuple[bool, str | None]:
        """Classify a square ROI as occupied or empty without discriminating piece color."""
        params = params or {
            "contrast_threshold": 12.0,
            "mean_diff_threshold": 6.0,
            "std_dev_threshold": 10.0,
            "white_margin": 14.0,
            "occupied_ratio": 0.08,
        }
        if roi.size == 0:
            return False, None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)
        h, w = gray.shape
        pad = max(3, min(h, w) // 10)

        corner_window = min(pad, h // 2, w // 2)
        corners = np.concatenate(
            [
                gray[:corner_window, :corner_window].ravel(),
                gray[:corner_window, -corner_window:].ravel(),
                gray[-corner_window:, :corner_window].ravel(),
                gray[-corner_window:, -corner_window:].ravel(),
            ]
        )
        background = float(np.median(corners)) if corners.size else 127.0

        center_margin = max(5, min(h, w) // 4)
        inner_y0, inner_y1 = center_margin, h - center_margin
        inner_x0, inner_x1 = center_margin, w - center_margin
        if inner_y1 <= inner_y0 or inner_x1 <= inner_x0:
            center = gray
        else:
            center = gray[inner_y0:inner_y1, inner_x0:inner_x1]
        if center.size == 0:
            center = gray

        diff = center.astype(np.float32) - background
        abs_diff = np.abs(diff)
        contrast_ratio = float(np.mean(abs_diff > params["contrast_threshold"]))
        mean_diff = float(np.mean(abs_diff))
        std_dev = float(np.std(center.astype(np.float32)))
        dark_ratio = float(np.mean(center < background - params["white_margin"]))
        light_ratio = float(np.mean(center > background + params["white_margin"]))

        occupied = (
            contrast_ratio > params["occupied_ratio"] and mean_diff > params["mean_diff_threshold"]
        ) or (
            std_dev > params["std_dev_threshold"] and contrast_ratio > params["occupied_ratio"] / 2.0
        ) or (
            dark_ratio > 0.08 or light_ratio > 0.08
        )

        if not occupied:
            return False, None

        return True, "occupied"

    def _score_expected_board(self, detections: list[dict], expected: dict[int, str]) -> float:
        """Score a set of detections against an expected board map."""
        score = 0.0
        matched = {item["square"]: item for item in detections}
        for square, expected_label in expected.items():
            found = matched.get(square)
            if found is None:
                score -= 4.0
                continue
            if found["label"] == expected_label or found["label"] == "occupied":
                score += 3.0
            else:
                score -= 2.0

        for item in detections:
            if item["square"] not in expected:
                score -= 3.0
        return score

    def auto_tune(
        self,
        frames: list[np.ndarray],
        expected: dict[int, str] | None = None,
        candidates: list[dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Choose the best threshold set for the current lighting using a known board state."""
        if not frames:
            return self.params

        expected_map = expected or {}
        candidates = candidates or _DEFAULT_TUNING_CANDIDATES

        best_params = self.params
        best_score = float("-inf")
        for params in candidates:
            total = 0.0
            for frame in frames:
                if self.board_detector is None or self.board_detector.H is None:
                    continue
                detections = self.detect(frame, params=params)
                mapped: list[dict] = []
                for detection in detections:
                    center_x = (detection.x1 + detection.x2) / 2.0
                    center_y = (detection.y1 + detection.y2) / 2.0
                    square = self.board_detector.to_square(center_x, center_y)
                    if square is None:
                        continue
                    mapped.append({"square": square, "label": detection.label})
                total += self._score_expected_board(mapped, expected_map)

            if total > best_score:
                best_score = total
                best_params = params

        self.params = best_params
        return best_params

    def detect(self, frame: np.ndarray, params: dict[str, float] | None = None) -> list[Detection]:
        """Detect occupied squares and their color from a calibrated board image."""
        if self.board_detector is None or self.board_detector.H is None:
            return []

        warped = self.board_detector.warped_view(frame)
        if warped is None:
            return []

        params = params or self.params
        detections: list[Detection] = []
        cell = self.board_detector.CELL
        for row in range(8):
            for col in range(8):
                x0, y0 = col * cell, row * cell
                x1, y1 = x0 + cell, y0 + cell
                roi = warped[y0:y1, x0:x1]
                occupied, label = self.classify_roi(roi, params=params)
                if not occupied or label is None:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)
                h, w = gray.shape
                pad = max(3, min(h, w) // 10)
                center = gray[pad : h - pad, pad : w - pad]
                if center.size == 0:
                    center = gray
                diff = np.abs(center.astype(np.float32) - float(np.median(gray)))
                contrast_ratio = float(np.mean(diff > params["contrast_threshold"]))
                mean_diff = float(np.mean(diff))
                std_dev = float(np.std(center.astype(np.float32)))
                conf = 0.45 * min(contrast_ratio / (params["occupied_ratio"] * 2.5), 1.0)
                conf += 0.35 * min(mean_diff / max(params["mean_diff_threshold"] * 2.0, 1.0), 1.0)
                conf += 0.20 * min(std_dev / max(params["std_dev_threshold"] * 1.8, 1.0), 1.0)
                conf = float(np.clip(conf, 0.0, 1.0))

                proj = self.board_detector.warp_to_frame(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    self.board_detector.Hinv,
                )
                pts = proj.astype(int)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)

                active_mask = np.abs(gray.astype(np.float32) - np.median(gray)) > params["contrast_threshold"]
                if active_mask.any():
                    ys, xs = np.where(active_mask)
                    local_cx = float(xs.mean())
                    local_cy = float(ys.mean())
                    point = self.board_detector.warp_to_frame([[x0 + local_cx, y0 + local_cy]], self.board_detector.Hinv)[0]
                    center_x, center_y = float(point[0]), float(point[1])
                else:
                    center_x = float((x_min + x_max) / 2.0)
                    center_y = float((y_min + y_max) / 2.0)

                square = self.board_detector.to_square(center_x, center_y)
                if square is None:
                    continue

                detections.append(
                    Detection(
                        class_id=0,
                        label=label,
                        conf=float(np.clip(conf, 0.0, 1.0)),
                        x1=int(x_min),
                        y1=int(y_min),
                        x2=int(x_max),
                        y2=int(y_max),
                        center=(center_x, center_y),
                    )
                )
        return detections

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw the occupancy boxes on a frame and return the modified frame."""
        for detection in detections:
            color = (0, 255, 0) if detection.label == "occupied" else (255, 255, 255)
            cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
            cv2.putText(
                frame,
                f"{detection.label} {detection.conf:.2f}",
                (detection.x1, max(detection.y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return frame
