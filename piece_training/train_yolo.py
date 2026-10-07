"""Training script for the YOLO chess-piece detector.

This module uses fixed runtime configuration values instead of CLI arguments so the
training setup stays stable and consistent across local runs.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


MODEL = "yolo11n.pt"
DATA = str(Path(__file__).resolve().with_name("data.yaml"))
EPOCHS = 100
IMGSZ = 640
BATCH = 2
DEVICE = "cpu"
PROJECT = str(Path(__file__).resolve().parent.parent / "runs")
NAME = "chess_train"
EXIST_OK = True
EXPORT_ONNX = False


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
    )

    if EXPORT_ONNX:
        model.export(format="onnx")

    print("\nTraining complete. Best weights:")
    print(Path(PROJECT) / NAME / "weights" / "best.pt")


if __name__ == "__main__":
    main()
