"""Training script for the YOLO chess-piece detector.

This module uses fixed runtime configuration values instead of CLI arguments so the
training setup stays stable and consistent across local runs.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


MODEL = "yolo11m.pt"
DATA = str(Path(__file__).resolve().with_name("data.yaml"))
EPOCHS = 180
IMGSZ = 640
BATCH = 8
DEVICE = "mps"
PROJECT = str(Path(__file__).resolve().parent.parent / "runs")
NAME = "chess_train_max"
EXIST_OK = True
EXPORT_ONNX = False

# Small chess-piece dataset + 13 classes: keep augmentations moderate so we do not
# distort piece silhouettes or make black/white pieces too similar.
AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 6,
    "translate": 0.08,
    "scale": 0.2,
    "shear": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mixup": 0.0,
    "mosaic": 0.0,
    "copy_paste": 0.0,
}


def main() -> None:
    """Run the training process with the module-level configuration.

    :return: None.
    """
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=EXIST_OK,
        patience=40,
        optimizer="AdamW",
        cos_lr=True,
        amp=True,
        **AUGMENTATION,
    )

    if EXPORT_ONNX:
        model.export(format="onnx")

    print("\nTraining complete. Best weights:")
    print(Path(PROJECT) / NAME / "weights" / "best.pt")


if __name__ == "__main__":
    main()
