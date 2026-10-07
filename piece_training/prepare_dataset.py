"""Convert Roboflow-style polygon annotations into YOLO bounding boxes.

The script prepares the dataset structure used by the chess-piece detector,
normalizes labels, and writes the final `data.yaml` file for training.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "Chess_pieces"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "trained"


def ensure_dataset_layout() -> None:
    """Create the expected train/valid/test directories for the prepared dataset.

    :return: None.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "valid", "test"):
        (OUTPUT_ROOT / split_name / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split_name / "labels").mkdir(parents=True, exist_ok=True)


def load_source_names() -> list[str]:
    """Read class names from the source dataset YAML file.

    :return: Ordered list of class names from the dataset configuration.
    """
    candidates = [
        SOURCE_ROOT / "data.yaml",
        SOURCE_ROOT / "data.yaml.yaml",
    ]
    source_yaml = next((candidate for candidate in candidates if candidate.exists()), None)
    if source_yaml is None:
        raise FileNotFoundError(
            f"Missing source dataset config: {SOURCE_ROOT / 'data.yaml'}\n"
            "Expected a source dataset under data/raw/Chess_pieces/ with a YOLO-style data.yaml."
        )

    names: dict[int, str] = {}
    in_names = False

    for raw_line in source_yaml.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line == "names:":
            in_names = True
            continue

        if in_names:
            match = re.match(r"^(\d+)\s*:\s*(.+?)\s*$", line)
            if match:
                names[int(match.group(1))] = match.group(2)
                continue

            if line.startswith(("train:", "val:", "test:")):
                break

    if not names:
        raise ValueError(f"Could not parse names from {source_yaml}")

    return [names[idx] for idx in sorted(names)]


def convert_polygon_to_yolo_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """Convert polygon annotation values into a YOLO bounding box.

    :param values: Eight values describing the polygon corners in order x1 y1 x2 y2 ...
    :return: A tuple of (center_x, center_y, width, height).
    """
    if len(values) != 8:
        raise ValueError(f"Expected 8 polygon values, got {len(values)}")

    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return x_center, y_center, width, height


def normalize_label_line(line: str) -> str | None:
    """Normalize one annotation line from polygon format into YOLO rectangle format.

    :param line: Raw annotation line from the source dataset.
    :return: A normalized YOLO label string or None when the line is invalid.
    """
    parts = line.strip().split()
    if not parts:
        return None

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    if len(coords) != 8:
        return None

    x_center, y_center, width, height = convert_polygon_to_yolo_bbox(coords)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def copy_and_convert_split(split_name: str, source_names: list[str]) -> None:
    """Copy one dataset split and convert all polygon annotations into YOLO labels.

    :param split_name: Split name such as 'train', 'valid', or 'test'.
    :param source_names: Ordered list of class names used by the dataset.
    :return: None.
    """
    source_split_dir = SOURCE_ROOT / split_name
    if not source_split_dir.exists():
        print(f"Skipping missing split: {source_split_dir}")
        return

    img_dir = source_split_dir / "images"
    label_dir = source_split_dir / "labels"
    target_dir = OUTPUT_ROOT / split_name
    target_img_dir = target_dir / "images"
    target_label_dir = target_dir / "labels"

    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print(f"No labels found in {label_dir}")

    for label_path in label_files:
        stem = label_path.stem
        image_match = next(img_dir.glob(f"{stem}.*"), None)
        if image_match is not None:
            shutil.copy2(image_match, target_img_dir / image_match.name)

        converted_lines = []
        for line in label_path.read_text().splitlines():
            normalized = normalize_label_line(line)
            if normalized is not None:
                converted_lines.append(normalized)

        target_label_path = target_label_dir / f"{stem}.txt"
        if converted_lines:
            target_label_path.write_text("\n".join(converted_lines) + "\n")
        else:
            target_label_path.write_text("")

    converted_count = len(list(target_label_dir.glob("*.txt")))
    print(f"Prepared {split_name}: {converted_count} labels copied to {target_dir}")


def write_yolo_config(names: list[str]) -> None:
    """Write the final `data.yaml` file that YOLO expects.

    :param names: Class names used for training.
    :return: None.
    """
    output_path = PROJECT_ROOT / "data.yaml"
    lines = [
        "train: data/trained/train/images",
        "val: data/trained/valid/images",
        "test: data/trained/test/images",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    for name in names:
        lines.append(f"  - {name}")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote dataset config: {output_path}")


def main() -> None:
    """Run dataset preparation for the chess-piece model.

    :return: None.
    """
    if not SOURCE_ROOT.exists():
        ensure_dataset_layout()
        raise SystemExit(
            f"Dataset source directory not found: {SOURCE_ROOT}\n"
            "Create the folder data/raw/Chess_pieces/ and place your YOLO dataset there, "
            "then run this script again."
        )

    source_names = load_source_names()
    for split_name in ("train", "valid", "test"):
        copy_and_convert_split(split_name, source_names)
    write_yolo_config(source_names)


if __name__ == "__main__":
    main()
