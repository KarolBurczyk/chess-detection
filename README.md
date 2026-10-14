# Chess AI — live chessboard detection

This repository contains the active version of the project for real-time chessboard recognition from a camera feed. The main logic is implemented in the `chess_ai` package and is built around two core components:

- board detection through corner estimation, perspective transformation, and 8x8 mapping,
- YOLO-based piece detection and assignment of detections to specific squares.

## Project goal

The goal is to build a simple but working pipeline that:

1. captures frames from a camera,
2. detects the chessboard edges,
3. maps the image to an 8x8 board plane,
4. detects pieces with a YOLO model,
5. assigns detections to squares,
6. visualizes the result in a live window and provides a base for further chess-state logic.

## Current architecture

- `chess_ai/` — active application code
  - `main.py` — live application entry point
  - `board_detector.py` — board detection, corner fitting, and board geometry
  - `piece_detector.py` — YOLO-based piece detector
  - `move_tracker.py` — legal move scoring and move confirmation logic
  - `board_state.py` — chess state wrapper and legal move handling
  - `board_view.py` — dedicated board visualization window
  - `config.py` — model and camera source configuration
  - `raw_detector.py` — quick prototype or debugging detector variant
- `piece_training/` — dataset preparation and YOLO training utilities
  - `prepare_dataset.py` — dataset conversion and export to YOLO labels
  - `train_yolo.py` — YOLO training script
- `data/` — single data folder split into raw and trained datasets
  - `data/raw/Chess_pieces/` — raw Kaggle source data
  - `data/trained/` — prepared training, validation, and test splits
- `runs/` — training outputs and model weights
- `yolo11n.pt` — base YOLO model file
- `requirements.txt` — project dependencies

## How to run

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Run the live detector

From the project root:

```bash
PYTHONPATH=. python3 -m chess_ai.main
```

Or directly from the `chess_ai` directory:

```bash
cd chess_ai
python3 main.py
```

The default capture source is configured in `chess_ai/config.py`:

```python
SOURCE = 0
```

If you want to use another source, replace it with a video path or another camera index.

## How the pipeline works

The project uses a YOLO model to detect board objects in each frame. Then:

- a homography is estimated between the camera image and the chessboard plane,
- detected piece centers are mapped to 8x8 square coordinates,
- board orientation and margins are estimated,
- detections are overlaid on the live frame with square labels and confidence values,
- the board state is updated and displayed in a separate live chessboard window.

## Training and dataset preparation

### Download the public Kaggle dataset

This project can use the public dataset from Kaggle:

https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset

Before running the script, make sure you have a Kaggle API token configured in `~/.kaggle/kaggle.json`.

Then run from the project root:

```bash
python3 piece_training/download_kaggle_dataset.py
```

The script downloads the raw dataset and stores it under:

```text
data/raw/Chess_pieces/
```

### Prepare the dataset for YOLO

After download:

```bash
python3 piece_training/prepare_dataset.py
```

This converts the raw source annotations into the YOLO-ready format and creates the prepared split folders under:

```text
data/trained/
```

### Train the model

```bash
cd piece_training
python3 train_yolo.py
```

## Visualizations

Below are sample screenshots from the live chessboard recognition workflow and the rendered board view.

<div align="center">

![Board detection preview](visualisation/Zrzut%20ekranu%202026-10-7%20o%2014.37.08.png)

![Detected board and piece overlay](visualisation/Zrzut%20ekranu%202026-10-7%20o%2014.37.24.png)

</div>

The training configuration is fixed in the script itself. The dataset paths and model settings are also defined in `chess_ai/config.py`.

## Current status

The project currently includes:

- stable camera capture and live board detection,
- perspective calibration and 8x8 board mapping,
- YOLO-based piece detection and assignment to squares,
- live visualization of the detected board and candidate moves,
- chess-state handling through `python-chess` in `board_state.py`.

This is a working prototype for practical board-position recognition from camera input. It already covers the main pipeline from frame capture to square-level interpretation and live rendering, while further robustness improvements are still ongoing.

## Next focus areas

The most relevant improvements in the near term are:

1. stabilizing detection under more camera angles and lighting conditions,
2. aggregating board state across multiple frames to reduce jitter,
3. improving move tracking and confirmation quality,
4. refining the UI and debugging overlays for easier live analysis.

## Repository files at a glance

- `chess_ai/main.py` — main live camera loop
- `chess_ai/board_detector.py` — geometry and board calibration
- `chess_ai/piece_detector.py` — YOLO inference wrapper
- `chess_ai/move_tracker.py` — candidate move tracking and confirmation logic
- `chess_ai/board_view.py` — live board preview rendering
- `piece_training/prepare_dataset.py` — dataset conversion utility
- `piece_training/train_yolo.py` — model training script
- `data/raw/` — raw source data
- `data/trained/` — prepared training data
- `runs/` — training runs and weight exports
