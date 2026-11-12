import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess_ai"))

from piece_detector import PieceDetector


def test_classify_roi_detects_black_occupied_cell():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 200, dtype=np.uint8)
    roi[25:55, 25:55] = 25

    occupied, color = detector.classify_roi(roi)

    assert occupied is True
    assert color == "black"


def test_classify_roi_detects_white_occupied_cell():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 30, dtype=np.uint8)
    roi[25:55, 25:55] = 220

    occupied, color = detector.classify_roi(roi)

    assert occupied is True
    assert color == "white"


def test_classify_roi_detects_white_piece_on_light_square():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 220, dtype=np.uint8)
    roi[22:58, 22:58] = 245

    occupied, color = detector.classify_roi(roi)

    assert occupied is True
    assert color == "white"


def test_classify_roi_detects_black_piece_on_dark_square():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 200, dtype=np.uint8)
    roi[22:58, 22:58] = 35

    occupied, color = detector.classify_roi(roi)

    assert occupied is True
    assert color == "black"
