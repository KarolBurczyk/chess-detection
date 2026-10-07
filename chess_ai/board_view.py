from __future__ import annotations

import cv2
import chess
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

LIGHT = (181, 217, 240)
DARK = (99, 136, 181)
BG = (30, 30, 30)
TEXT = (230, 230, 230)
LAST_MOVE = (60, 220, 240)
CANDIDATE = (220, 160, 60)
CHECK = (60, 60, 230)


class BoardView:
    """Render a chess board in a separate OpenCV window."""

    def __init__(self, square: int = 64, panel_width: int = 240) -> None:
        """Create the board renderer.

        :param square: Pixel size of one board square.
        :param panel_width: Width of the side panel with status text.
        :return: None.
        """
        self.SQ = square
        self.M = 24
        self.PANEL = panel_width
        self.flipped = False

    def toggle_flip(self) -> None:
        """Invert board orientation between white-at-bottom and black-at-bottom.

        :param self: The board view instance.
        :return: None.
        """
        self.flipped = not self.flipped

    def _xy(self, square_index: int) -> tuple[int, int]:
        """Return the top-left pixel position of a square inside the board image.

        :param square_index: Square index in python-chess format.
        :return: Pixel coordinates for the square's top-left corner.
        """
        file = chess.square_file(square_index)
        rank = chess.square_rank(square_index)
        col, row = (7 - file, rank) if self.flipped else (file, 7 - rank)
        return self.M + col * self.SQ, self.M + row * self.SQ

    def _tint(self, img: np.ndarray, square_index: int, color: tuple[int, int, int], alpha: float) -> None:
        """Overlay a transparent color on a single square.

        :param img: The full board image.
        :param square_index: Square index to tint.
        :param color: BGR color tuple.
        :param alpha: Overlay transparency in range 0..1.
        :return: None.
        """
        x, y = self._xy(square_index)
        roi = img[y : y + self.SQ, x : x + self.SQ]
        overlay = np.full_like(roi, color)
        img[y : y + self.SQ, x : x + self.SQ] = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)

    def _draw_piece(self, img: np.ndarray, square_index: int, piece: chess.Piece) -> None:
        """Draw a single chess piece as a circle with a letter.

        :param img: The board image to draw into.
        :param square_index: Square where the piece stands.
        :param piece: Piece object from python-chess.
        :return: None.
        """
        x, y = self._xy(square_index)
        cx, cy = x + self.SQ // 2, y + self.SQ // 2
        radius = int(self.SQ * 0.38)
        white = piece.color == chess.WHITE
        fill = (245, 245, 245) if white else (40, 40, 40)
        edge = (40, 40, 40) if white else (230, 230, 230)
        ink = (30, 30, 30) if white else (240, 240, 240)

        cv2.circle(img, (cx, cy), radius, fill, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, edge, 2, cv2.LINE_AA)

        letter = piece.symbol().upper()
        (tw, th), _ = cv2.getTextSize(letter, FONT, 0.9, 2)
        cv2.putText(img, letter, (cx - tw // 2, cy + th // 2), FONT, 0.9, ink, 2, cv2.LINE_AA)

    def render(self, board: chess.Board, history: list[str], candidate: str | None = None) -> np.ndarray:
        """Render the board into a numpy image for OpenCV display.

        :param board: Current chess board state to render.
        :param history: List of already played moves.
        :param candidate: Optional candidate move currently suspected by the tracker.
        :return: A BGR image of the rendered board.
        """
        square_size, margin = self.SQ, self.M
        height = 2 * margin + 8 * square_size
        width = margin + 8 * square_size + self.PANEL
        img = np.full((height, width, 3), BG, dtype=np.uint8)

        for square_index in chess.SQUARES:
            x, y = self._xy(square_index)
            light = (chess.square_file(square_index) + chess.square_rank(square_index)) % 2 == 1
            cv2.rectangle(img, (x, y), (x + square_size - 1, y + square_size - 1), LIGHT if light else DARK, -1)

        if board.move_stack:
            last_move = board.move_stack[-1]
            self._tint(img, last_move.from_square, LAST_MOVE, 0.55)
            self._tint(img, last_move.to_square, LAST_MOVE, 0.55)
        if candidate:
            try:
                move = chess.Move.from_uci(candidate)
                self._tint(img, move.from_square, CANDIDATE, 0.55)
                self._tint(img, move.to_square, CANDIDATE, 0.55)
            except ValueError:
                pass
        if board.is_check():
            king_square = board.king(board.turn)
            if king_square is not None:
                self._tint(img, king_square, CHECK, 0.65)

        for square_index, piece in board.piece_map().items():
            self._draw_piece(img, square_index, piece)

        for i in range(8):
            file_index = 7 - i if self.flipped else i
            rank_index = i if self.flipped else 7 - i
            cv2.putText(img, "abcdefgh"[file_index], (margin + i * square_size + square_size // 2 - 5, margin + 8 * square_size + 17), FONT, 0.5, TEXT, 1, cv2.LINE_AA)
            cv2.putText(img, str(rank_index + 1), (7, margin + i * square_size + square_size // 2 + 5), FONT, 0.5, TEXT, 1, cv2.LINE_AA)

        self._draw_panel(img, board, history, candidate)
        return img

    def _draw_panel(self, img: np.ndarray, board: chess.Board, history: list[str], candidate: str | None) -> None:
        """Draw the side panel with turn, check and move list.

        :param img: Full board image to annotate.
        :param board: Current board state.
        :param history: List of SAN moves already played.
        :param candidate: Optional candidate move string.
        :return: None.
        """
        x0 = self.M + 8 * self.SQ + 14
        y = 34
        line_h = 22

        def put(text: str, color: tuple[int, int, int] = TEXT, scale: float = 0.6, bold: int = 1) -> None:
            nonlocal y
            cv2.putText(img, text, (x0, y), FONT, scale, color, bold, cv2.LINE_AA)
            y += line_h

        put(f"Tura: {'biale' if board.turn == chess.WHITE else 'czarne'}", (255, 255, 255), 0.7, 2)

        if board.is_checkmate():
            put(f"Mat! Wynik: {board.result()}", (80, 80, 255), 0.6, 2)
        elif board.is_game_over():
            put(f"Koniec: {board.result()}", (80, 80, 255), 0.6, 2)
        elif board.is_check():
            put("Szach!", (80, 80, 255), 0.6, 2)

        if candidate:
            put(f"Rozpoznaje: {candidate}", CANDIDATE, 0.55)

        y += 8
        put("Ruchy:", (255, 255, 255), 0.6, 2)

        rows: list[str] = []
        for index in range(0, len(history), 2):
            black_move = history[index + 1] if index + 1 < len(history) else ""
            rows.append(f"{index // 2 + 1}. {history[index]}   {black_move}")

        max_rows = max(1, (img.shape[0] - y - 10) // 20)
        for row in rows[-max_rows:]:
            cv2.putText(img, row, (x0, y), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
            y += 20
