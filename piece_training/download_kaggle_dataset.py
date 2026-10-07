"""Download the Kaggle chess-piece dataset using the KaggleHub API.

Example from Kaggle:
    path = kagglehub.dataset_download("imtkaggleteam/chess-pieces-detection-image-dataset")
"""

from __future__ import annotations

from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_SLUG = "imtkaggleteam/chess-pieces-detection-image-dataset"
RAW_SOURCE = PROJECT_ROOT / "data" / "raw" / "Chess_pieces"


def main() -> None:
    """Download the dataset into the project data/raw folder.

    :return: None.
    """
    RAW_SOURCE.mkdir(parents=True, exist_ok=True)
    dataset_path = kagglehub.dataset_download(DATASET_SLUG)
    if dataset_path:
        if RAW_SOURCE.exists():
            import shutil

            for child in RAW_SOURCE.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        shutil.copytree(Path(dataset_path), RAW_SOURCE, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
