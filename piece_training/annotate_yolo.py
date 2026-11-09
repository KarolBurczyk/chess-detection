#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_ai.board_detector import BoardDetector

IMAGES_DIR = "/Users/karol/Desktop/new_chess/data/annotate/images"
LABELS_DIR = "/Users/karol/Desktop/new_chess/data/annotate/labels"
MODEL_PATH = "/Users/karol/Desktop/new_chess/runs/detect/runs/chess_train/weights/best.pt"
ENABLE_PRELABELING = True
FONT_SCALE = 0.9
LABEL_THICKNESS = 2


def _load_dataset_classes() -> list[str]:
    config_path = Path(__file__).resolve().with_name("data.yaml")
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            names = data.get("names") or []
            if names:
                return [str(name) for name in names]
        except Exception:
            pass

    return [
        "bishop",
        "black-bishop",
        "black-king",
        "black-knight",
        "black-pawn",
        "black-queen",
        "black-rook",
        "white-bishop",
        "white-king",
        "white-knight",
        "white-pawn",
        "white-queen",
        "white-rook",
    ]


CLASSES = _load_dataset_classes()


class BoxAnnotator:
    def __init__(self, images_dir: str, labels_dir: str, classes: list[str]) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes = classes
        self.class_map = {name: idx for idx, name in enumerate(classes)}
        self.model = self._load_model()
        self.board_detector = BoardDetector()
        self.image_paths = self._discover_image_paths()
        self.index = 0
        self.current_boxes: list[dict[str, Any]] = []
        self.img = None
        self.model_input = None
        self.img_path: Path | None = None
        self.selected_idx: int | None = None
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_end = (0, 0)
        self.current_class_idx = 0
        self.pending_replace = False
        self.resize_mode: str | None = None
        self.resize_origin: dict[str, Any] | None = None
        self.resize_initial: tuple[int, int] | None = None
        self.preview_box: dict[str, int] | None = None
        self.draw_mode = False
        self.select_mode = True

    def _load_model(self) -> YOLO | None:
        if not ENABLE_PRELABELING:
            return None
        if not Path(MODEL_PATH).exists():
            print(f"Pre-label model not found: {MODEL_PATH}")
            return None
        try:
            model = YOLO(MODEL_PATH)
            print(f"Loaded pre-label model: {MODEL_PATH}")
            return model
        except Exception as exc:  # pragma: no cover - user-facing debug output
            print(f"Could not load pre-label model: {exc}")
            return None

    def _read_image_file(self, path: Path) -> np.ndarray | None:
        suffix = path.suffix.lower()
        if suffix in {".heic", ".heif"}:
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except Exception:
                pass
            try:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        return cv2.imread(str(path))

    def _discover_image_paths(self) -> list[Path]:
        image_paths: list[Path] = []
        for path in sorted(self.images_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}:
                continue
            if self._read_image_file(path) is not None:
                image_paths.append(path)
            else:
                print(f"Skipping unreadable image: {path}")
        return image_paths

    def _prepare_detection_image(self, frame: np.ndarray) -> np.ndarray:
        board = self.board_detector
        corners = board.detect_board_corners(frame)
        if corners is None:
            return frame

        board.board_corners = np.median(np.stack([corners]), axis=0).astype(np.float32)
        H, Hinv, margins, score = board.calibrate(board.board_corners, [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)])
        if score < board.MIN_FIT:
            return frame

        board.H = H
        board.Hinv = Hinv
        board.margins = margins
        board.fit_score = score
        return cv2.warpPerspective(frame, H, (board.WARP, board.WARP))

    def _convert_prediction_to_image_coords(self, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
        if self.img is None or self.model_input is None or self.board_detector.Hinv is None:
            return int(x1), int(y1), int(x2), int(y2)

        pts = np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.board_detector.Hinv).reshape(-1, 2)
        xs = mapped[:, 0]
        ys = mapped[:, 1]
        img_h, img_w = self.img.shape[:2]
        return (
            int(np.clip(xs.min(), 0, img_w - 1)),
            int(np.clip(ys.min(), 0, img_h - 1)),
            int(np.clip(xs.max(), 0, img_w - 1)),
            int(np.clip(ys.max(), 0, img_h - 1)),
        )

    def _load_image(self, image_index: int) -> None:
        if not self.image_paths:
            raise FileNotFoundError(
                f"No readable images found in {self.images_dir}. "
                "Check that the files are valid JPG/PNG/HEIC images and not corrupted or empty."
            )
        self.index = max(0, min(image_index, len(self.image_paths) - 1))
        self.img_path = self.image_paths[self.index]
        raw_image = self._read_image_file(self.img_path)
        if raw_image is None:
            raise ValueError(f"Could not read image: {self.img_path}")
        self.img = raw_image
        self.model_input = self._prepare_detection_image(raw_image)
        self.current_boxes = self._read_existing_boxes(self.img_path)
        if not self.current_boxes and self.model is not None:
            self._apply_model_predictions()
        self.selected_idx = None
        self.resize_mode = None
        self.resize_origin = None
        self.resize_initial = None

    def _apply_model_predictions(self) -> None:
        if self.img is None or self.model is None:
            return
        detection_image = self.model_input if self.model_input is not None else self.img
        try:
            result = self.model(detection_image, conf=0.2, iou=0.5, agnostic_nms=True, verbose=False)[0]
        except Exception as exc:  # pragma: no cover - model runtime failure should be non-fatal
            print(f"Pre-label prediction failed: {exc}")
            return

        boxes: list[dict[str, Any]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            if cls_id >= len(self.classes):
                continue
            if self.model_input is not None and self.model_input is not self.img and self.board_detector.Hinv is not None:
                x1, y1, x2, y2 = self._convert_prediction_to_image_coords(x1, y1, x2, y2)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            boxes.append({
                "class_id": cls_id,
                "x1": max(0, min(x1, self.img.shape[1])),
                "y1": max(0, min(y1, self.img.shape[0])),
                "x2": max(0, min(x2, self.img.shape[1])),
                "y2": max(0, min(y2, self.img.shape[0])),
            })
        self.current_boxes = boxes
        if boxes:
            self.selected_idx = 0
        print(f"Applied {len(boxes)} pre-label boxes from current model.")

    def _read_existing_boxes(self, image_path: Path) -> list[dict[str, Any]]:
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes: list[dict[str, Any]] = []
        if not label_path.exists():
            return boxes

        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            try:
                cls_id = int(tokens[0])
                cx, cy, w, h = map(float, tokens[1:5])
            except ValueError:
                continue
            image_w = self.img.shape[1] if self.img is not None else 1
            image_h = self.img.shape[0] if self.img is not None else 1
            x1 = (cx - w / 2.0) * image_w
            y1 = (cy - h / 2.0) * image_h
            x2 = (cx + w / 2.0) * image_w
            y2 = (cy + h / 2.0) * image_h
            boxes.append({
                "class_id": cls_id,
                "x1": int(min(max(x1, 0), image_w)),
                "y1": int(min(max(y1, 0), image_h)),
                "x2": int(min(max(x2, 0), image_w)),
                "y2": int(min(max(y2, 0), image_h)),
            })
        return boxes

    def _draw_box(self, image: np.ndarray, box: dict[str, Any], color: tuple[int, int, int], selected: bool = False) -> None:
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        line_width = 2 if not selected else 4
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        label = self.classes[box["class_id"]]
        cv2.putText(
            image,
            f"{label}",
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            LABEL_THICKNESS,
        )
        if selected:
            for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.circle(image, (px, py), 6, (255, 255, 255), -1)
                cv2.circle(image, (px, py), 4, (0, 0, 255), -1)

    def _nearest_box(self, x: int, y: int) -> int | None:
        best_idx: int | None = None
        best_dist = 999999.0
        for idx, box in enumerate(self.current_boxes):
            x1 = min(box["x1"], box["x2"])
            y1 = min(box["y1"], box["y2"])
            x2 = max(box["x1"], box["x2"])
            y2 = max(box["y1"], box["y2"])

            inside = x1 <= x <= x2 and y1 <= y <= y2
            if inside:
                return idx

            edge_dist = min(
                abs(x - x1),
                abs(x - x2),
                abs(y - y1),
                abs(y - y2),
            )
            if edge_dist < best_dist:
                best_dist = edge_dist
                best_idx = idx

        if best_idx is not None and best_dist < 18:
            return best_idx
        return None

    def _resize_handle(self, x: int, y: int, box: dict[str, Any]) -> str | None:
        handles = {
            "tl": ((box["x1"], box["y1"]), 12),
            "tr": ((box["x2"], box["y1"]), 12),
            "bl": ((box["x1"], box["y2"]), 12),
            "br": ((box["x2"], box["y2"]), 12),
        }
        for handle_name, ((hx, hy), radius) in handles.items():
            if (x - hx) ** 2 + (y - hy) ** 2 <= radius ** 2:
                return handle_name
        return None

    def _set_class_for_current_selection(self, class_idx: int) -> None:
        if self.selected_idx is not None:
            self.current_boxes[self.selected_idx]["class_id"] = class_idx
            print(f"Updated selected box class: {self.classes[class_idx]}")
        else:
            self.current_class_idx = class_idx
            print(f"Selected active class: {self.classes[class_idx]}")

    def _save_current_image(self) -> None:
        if self.img_path is None:
            return
        labels_path = self.labels_dir / f"{self.img_path.stem}.txt"
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        out_lines: list[str] = []
        for box in self.current_boxes:
            class_id = int(box["class_id"])
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            img_h, img_w = self.img.shape[:2]
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            out_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        labels_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    def _draw_current_state(self) -> None:
        if self.img is None:
            return
        display = self.img.copy()
        for idx, box in enumerate(self.current_boxes):
            self._draw_box(display, box, (0, 0, 255) if idx == self.selected_idx else (0, 255, 0), selected=(idx == self.selected_idx))

        if self.dragging:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
            if self.resize_mode is None:
                self.preview_box = {
                    "x1": min(x1, x2),
                    "y1": min(y1, y2),
                    "x2": max(x1, x2),
                    "y2": max(y1, y2),
                }
                cv2.putText(
                    display,
                    "Preview new box",
                    (max(10, min(x1, x2)), max(16, min(y1, y2) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

        instructions = [
            "Controls:",
            "- v = select mode",
            "- b = draw mode",
            "- left-click = select box / draw when in draw mode",
            "- drag selected corner = resize box",
            "- Delete = remove selected box",
            "- 0-9/a/b = change selected class",
            "- m = prelabel current image",
            "- s = save, n/p = next/prev, q = quit",
        ]
        y = 36
        for line in instructions:
            cv2.putText(
                display,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                (255, 255, 255),
                2,
            )
            y += 28

        if self.draw_mode:
            cv2.putText(
                display,
                "DRAW MODE ACTIVE",
                (12, display.shape[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 165, 255),
                2,
            )
        elif self.select_mode:
            cv2.putText(
                display,
                "SELECT MODE ACTIVE",
                (12, display.shape[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            display,
            f"Image {self.index + 1}/{len(self.image_paths)} | class: {self.classes[self.current_class_idx]} | press q quit",
            (10, display.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        if self.selected_idx is not None:
            label = self.classes[self.current_boxes[self.selected_idx]["class_id"]]
            cv2.putText(
                display,
                f"Selected: {label}",
                (10, display.shape[0] - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        cv2.imshow("YOLO box annotator", display)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if self.img is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selected_idx is not None:
                box = self.current_boxes[self.selected_idx]
                handle = self._resize_handle(x, y, box)
                if handle is not None:
                    self.dragging = True
                    self.resize_mode = handle
                    self.resize_origin = box.copy()
                    self.resize_initial = (x, y)
                    print(f"Resize handle {handle} started")
                    return

            if self.draw_mode:
                self.selected_idx = None
                self.dragging = True
                self.resize_mode = None
                self.drag_start = (x, y)
                self.drag_end = (x, y)
                print("Draw mode: new box drag started")
                return

            idx = self._nearest_box(x, y)
            if idx is not None:
                self.selected_idx = idx
                print(f"Selected box #{idx}: {self.classes[self.current_boxes[idx]['class_id']]}")
                self.dragging = False
                self.drag_start = (0, 0)
                self.drag_end = (0, 0)
                return

            self.selected_idx = None
            self.dragging = True
            self.resize_mode = None
            self.drag_start = (x, y)
            self.drag_end = (x, y)
            print("New box drag started")

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.resize_mode is not None and self.selected_idx is not None and self.resize_origin is not None:
                base = self.resize_origin
                dx = x - self.resize_initial[0]
                dy = y - self.resize_initial[1]
                new_box = base.copy()
                if self.resize_mode == "tl":
                    new_box["x1"] = max(0, min(base["x1"] + dx, base["x2"] - 2))
                    new_box["y1"] = max(0, min(base["y1"] + dy, base["y2"] - 2))
                elif self.resize_mode == "tr":
                    new_box["x2"] = max(0, max(base["x1"] + 2, base["x2"] + dx))
                    new_box["y1"] = max(0, min(base["y1"] + dy, base["y2"] - 2))
                elif self.resize_mode == "bl":
                    new_box["x1"] = max(0, min(base["x1"] + dx, base["x2"] - 2))
                    new_box["y2"] = max(0, max(base["y1"] + 2, base["y2"] + dy))
                elif self.resize_mode == "br":
                    new_box["x2"] = max(0, max(base["x1"] + 2, base["x2"] + dx))
                    new_box["y2"] = max(0, max(base["y1"] + 2, base["y2"] + dy))
                self.current_boxes[self.selected_idx] = new_box
                self.drag_end = (x, y)
            else:
                self.drag_end = (x, y)
                if self.selected_idx is None:
                    self.preview_box = {
                        "x1": min(self.drag_start[0], x),
                        "y1": min(self.drag_start[1], y),
                        "x2": max(self.drag_start[0], x),
                        "y2": max(self.drag_start[1], y),
                    }

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            if self.resize_mode is not None and self.selected_idx is not None:
                self.resize_mode = None
                self.resize_origin = None
                self.resize_initial = None
                self.dragging = False
                self.drag_start = (0, 0)
                self.drag_end = (0, 0)
                return

            self.dragging = False
            x1 = min(self.drag_start[0], self.drag_end[0])
            y1 = min(self.drag_start[1], self.drag_end[1])
            x2 = max(self.drag_start[0], self.drag_end[0])
            y2 = max(self.drag_start[1], self.drag_end[1])
            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                box = {
                    "class_id": self.current_class_idx,
                    "x1": max(0, x1),
                    "y1": max(0, y1),
                    "x2": min(self.img.shape[1], x2),
                    "y2": min(self.img.shape[0], y2),
                }
                if self.pending_replace and self.selected_idx is not None:
                    self.current_boxes[self.selected_idx] = box
                    self.pending_replace = False
                else:
                    self.current_boxes.append(box)
                    self.selected_idx = len(self.current_boxes) - 1
                    print(f"Created box #{self.selected_idx}: {self.classes[self.current_class_idx]}")
            self.drag_start = (0, 0)
            self.drag_end = (0, 0)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_idx = None
            print("Selection cleared")

    def run(self) -> None:
        cv2.namedWindow("YOLO box annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("YOLO box annotator", self._on_mouse)

        if not self.image_paths:
            print(f"No images found in {self.images_dir}")
            return

        print("Controls: left-click = select box, left-drag on empty area = draw new box, drag selected box corners = resize, right-click = clear selection")
        print("m = apply current model, s = save labels, Delete = delete selected, e = replace selected box, 0-9 = class selection, a = black queen, b = black king, q = quit")
        print("When a box is selected, 0-9/a/b change its class only.")

        self._load_image(0)
        while True:
            self._draw_current_state()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):
                self._save_current_image()
                self._load_image(self.index + 1)
            elif key == ord('p'):
                self._save_current_image()
                self._load_image(self.index - 1)
            elif key == ord('s'):
                self._save_current_image()
                print(f"Saved: {self.img_path}")
            elif key in (ord('d'), 127, 8):
                if self.selected_idx is not None:
                    self.current_boxes.pop(self.selected_idx)
                    self.selected_idx = None
                    print("Deleted selected box")
            elif key == ord('e'):
                if self.selected_idx is not None:
                    self.pending_replace = True
                    print(f"Replace mode on: box #{self.selected_idx}")
            elif key == ord('c'):
                self.current_boxes.clear()
            elif key == ord('v'):
                self.draw_mode = False
                self.select_mode = True
                print("Select mode enabled")
            elif key == ord('b'):
                self.draw_mode = True
                self.select_mode = False
                print("Draw mode enabled")
            elif key == ord('m'):
                self._apply_model_predictions()
            elif key == ord('r'):
                self._load_image(self.index)
            elif key in range(48, 58):
                idx = key - 48
                if idx < len(self.classes):
                    self._set_class_for_current_selection(idx)
            elif key == ord('a'):
                idx = 10
                if idx < len(self.classes):
                    self._set_class_for_current_selection(idx)
            elif key == ord('k'):
                idx = 11
                if idx < len(self.classes):
                    self._set_class_for_current_selection(idx)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)
    annotator = BoxAnnotator(IMAGES_DIR, LABELS_DIR, CLASSES)
    annotator.run()
