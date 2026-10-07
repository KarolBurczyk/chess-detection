from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for chess piece detection")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model to start from")
    parser.add_argument("--data", default="data.yaml", help="Path to YAML dataset config")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="0", help="Device: cpu or 0 for CUDA/GPU")
    parser.add_argument("--project", default="runs", help="Output project dir")
    parser.add_argument("--name", default="train", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwrite existing run")
    parser.add_argument("--export", action="store_true", help="Export final model to ONNX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    if args.export:
        model.export(format="onnx")

    print("\nTraining complete. Best weights:")
    print(Path(args.project) / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
