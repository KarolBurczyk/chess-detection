from __future__ import annotations

import math
import re
from typing import Any

import chess

from board_detector import BoardDetector
from board_state import BoardState

_PIECES: dict[str, str] = {
    "pawn": "p",
    "rook": "r",
    "knight": "n",
    "horse": "n",
    "bishop": "b",
    "queen": "q",
    "king": "k",
}
_WHITE = {"white", "w"}
_BLACK = {"black", "b"}


def label_to_symbol(label: str) -> str | None:
    """Convert a YOLO label to a python-chess piece symbol.

    :param label: Label read from the detector, for example "white-rook" or "bP".
    :return: A chess piece symbol such as "R" or "p", or None if the label is not recognized.
    """
    raw = label.strip()
    if len(raw) == 1 and raw in "pnbrqkPNBRQK":
        return raw

    color: str | None = None
    piece: str | None = None
    for token in re.split(r"[\s_\-]+", raw.lower()):
        if token in _WHITE:
            color = "w"
        elif token in _BLACK:
            color = "b"
        elif token in _PIECES:
            piece = _PIECES[token]

    if color is None or piece is None:
        match = re.fullmatch(r"([wb])([pnbrqk])", raw.lower())
        if match:
            color, piece = match.group(1), match.group(2)

    if color is None or piece is None:
        return None
    return piece.upper() if color == "w" else piece


class MoveTracker:
    """Score legal moves against the detected board state and confirm one move at a time."""

    def __init__(
        self,
        state: BoardState,
        board_detector: BoardDetector,
        class_names: dict[int, str] | list[str],
        label_overrides: dict[str, str] | None = None,
        confirm_frames: int = 8,
        min_delta: float = 3.0,
        ambiguity: float = 1.0,
        eps: float = 0.05,
        base_fraction: float = 0.15,
    ) -> None:
        """Create the tracker for a board and detector pair.

        :param state: The current chess state that tracks the legal game position.
        :param board_detector: The board geometry object that converts pixels to squares.
        :param class_names: Mapping or list of YOLO class labels.
        :param label_overrides: Optional explicit mapping for labels that need custom conversion.
        :param confirm_frames: Number of consecutive frames that must agree on the same move.
        :param min_delta: Minimum score gap required before accepting a candidate move.
        :param ambiguity: Minimum gap required between the best and second-best move.
        :param eps: Small probability floor used to avoid log(0).
        :param base_fraction: Fraction from the bottom of the detection box used to estimate the square center.
        :return: None.
        """
        self.state = state
        self.board_detector = board_detector
        self.confirm_frames = confirm_frames
        self.min_delta = min_delta
        self.ambiguity = ambiguity
        self.eps = eps
        self.base_fraction = base_fraction

        self.class_map = self._build_class_map(class_names, label_overrides or {})

        self.candidate: str | None = None
        self.streak: int = 0
        self.history: list[str] = []
        self.match: tuple[int, int] = (0, 0)
        self.last_delta: float = 0.0

    @staticmethod
    def _build_class_map(class_names: dict[int, str] | list[str], overrides: dict[str, str]) -> dict[int, str]:
        """Translate model labels to python-chess piece symbols.

        :param class_names: Mapping or list of class names from the detector.
        :param overrides: Optional overrides for labels that should map to a custom symbol.
        :return: Mapping from YOLO class id to chess piece symbol.
        """
        items = class_names.items() if isinstance(class_names, dict) else enumerate(class_names)
        class_map: dict[int, str] = {}
        unknown: list[str] = []
        for cls_id, label in items:
            symbol = overrides.get(label) or label_to_symbol(label)
            if symbol is None:
                unknown.append(label)
            else:
                class_map[int(cls_id)] = symbol
        if unknown:
            print(f"[MoveTracker] Ignoring classes that were not recognized as chess pieces: {unknown}")
        if not class_map:
            raise ValueError(
                "No model class was recognized as a chess piece. "
                "Provide label_overrides, e.g. {'my-class': 'R'} (uppercase = white)."
            )
        return class_map

    def reset(self) -> None:
        """Reset the tracker and go back to the initial chess position.

        :param self: The tracker instance.
        :return: None.
        """
        self.state.board.reset()
        self.candidate, self.streak = None, 0
        self.history.clear()

    def observe(self, detections: list[Any]) -> dict[int, dict[str, float]]:
        """Map detected pieces to board squares and keep the strongest confidence for each square.

        :param detections: List of detection objects produced by the YOLO detector.
        :return: Mapping from square index to piece symbol with the highest confidence.
        """
        observations: dict[int, dict[str, float]] = {}
        for detection in detections:
            symbol = self.class_map.get(detection.class_id)
            if symbol is None:
                continue
            center_x = (detection.x1 + detection.x2) / 2.0
            center_y = detection.y2 - self.base_fraction * (detection.y2 - detection.y1)
            square = self.board_detector.to_square(center_x, center_y)
            if square is None:
                continue
            cell = observations.setdefault(square, {})
            cell[symbol] = max(cell.get(symbol, 0.0), detection.conf)
        return observations

    def _loglik(self, square: int, piece: chess.Piece | None, observations: dict[int, dict[str, float]]) -> float:
        """Compute a likelihood contribution for a piece on a square.

        :param square: Board square index.
        :param piece: Chess piece currently expected on the square, or None for empty square.
        :param observations: Detection confidence map by square.
        :return: Log-likelihood contribution for this piece-square pairing.
        """
        seen = observations.get(square)
        if piece is None:
            if not seen:
                return 0.0
            return math.log(max(1.0 - max(seen.values()), self.eps))
        probability = seen.get(piece.symbol(), 0.0) if seen else 0.0
        return math.log(max(probability, self.eps))

    def score_moves(self, observations: dict[int, dict[str, float]]) -> list[tuple[float, chess.Move]]:
        """Score all legal moves against the current detection evidence.

        :param observations: Observed squares and their confidence values.
        :return: List of scored moves sorted from best to worst.
        """
        board = self.state.board
        before = board.piece_map()
        scored: list[tuple[float, chess.Move]] = []
        for move in list(board.legal_moves):
            board.push(move)
            after = board.piece_map()
            board.pop()
            changed = [square for square in before.keys() | after.keys() if before.get(square) != after.get(square)]
            delta = sum(
                self._loglik(square, after.get(square), observations) - self._loglik(square, before.get(square), observations)
                for square in changed
            )
            scored.append((delta, move))
        scored.sort(key=lambda item: -item[0])
        return scored

    def _update_match(self, observations: dict[int, dict[str, float]]) -> None:
        """Update the match summary between expected pieces and detected pieces.

        :param observations: Observed piece probability map per square.
        :return: None.
        """
        expected = self.state.board.piece_map()
        hits = sum(1 for square, piece in expected.items() if observations.get(square, {}).get(piece.symbol(), 0.0) >= 0.3)
        self.match = (hits, len(expected))

    def update(self, detections: list[Any]) -> str | None:
        """Evaluate the current detections and confirm one move if it is stable.

        :param detections: List of detection objects from the current frame.
        :return: SAN notation of the confirmed move, or None if no move is accepted yet.
        """
        observations = self.observe(detections)
        self._update_match(observations)

        if self.state.board.is_game_over():
            return None

        scored = self.score_moves(observations)
        if not scored:
            return None

        best_delta, best_move = scored[0]
        second = scored[1][0] if len(scored) > 1 else float("-inf")
        self.last_delta = best_delta

        if best_delta < self.min_delta or best_delta - second < self.ambiguity:
            self.candidate, self.streak = None, 0
            return None

        uci = best_move.uci()
        if uci == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = uci, 1

        if self.streak >= self.confirm_frames:
            san = self.state.board.san(best_move)
            if self.state.make_move(uci):
                self.history.append(san)
                self.candidate, self.streak = None, 0
                return san
        return None
