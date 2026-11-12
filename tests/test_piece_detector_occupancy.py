import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess_ai"))

from board_state import BoardState
from move_tracker import MoveTracker
from piece_detector import PieceDetector


def test_classify_roi_detects_occupied_cell_on_dark_background():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 200, dtype=np.uint8)
    roi[25:55, 25:55] = 25

    occupied, label = detector.classify_roi(roi)

    assert occupied is True
    assert label == "occupied"


def test_classify_roi_detects_occupied_cell_on_light_background():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 30, dtype=np.uint8)
    roi[25:55, 25:55] = 220

    occupied, label = detector.classify_roi(roi)

    assert occupied is True
    assert label == "occupied"


def test_classify_roi_detects_occupied_piece_on_light_square():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 220, dtype=np.uint8)
    roi[22:58, 22:58] = 245

    occupied, label = detector.classify_roi(roi)

    assert occupied is True
    assert label == "occupied"


def test_classify_roi_detects_occupied_piece_on_dark_square():
    detector = PieceDetector()
    roi = np.full((80, 80, 3), 200, dtype=np.uint8)
    roi[22:58, 22:58] = 35

    occupied, label = detector.classify_roi(roi)

    assert occupied is True
    assert label == "occupied"


def test_move_tracker_counts_generic_occupied_detection_for_all_pieces():
    state = BoardState()
    tracker = MoveTracker(state, board_detector=None, class_names={0: "occupied"})
    observations = {square: {"P": 0.9} for square in state.board.piece_map()}

    tracker._update_match(observations)

    assert tracker.match == (len(state.board.piece_map()), len(state.board.piece_map()))


def test_move_tracker_scores_capture_when_target_square_stays_occupied():
    import chess

    tracker = MoveTracker(BoardState(), board_detector=None, class_names={0: "occupied"})
    board = tracker.state.board
    board.clear()
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("P"))
    board.set_piece_at(chess.D5, chess.Piece.from_symbol("p"))
    board.turn = chess.WHITE

    observations = {
        chess.E4: {"P": 0.95},
        chess.D5: {"P": 0.95},
    }

    scored = tracker.score_moves(observations)

    assert scored
    assert scored[0][1].uci() == "e4d5"
