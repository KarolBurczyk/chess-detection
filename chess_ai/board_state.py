import chess


class BoardState:
    """Convenience wrapper around a python-chess board state."""

    def __init__(self) -> None:
        """Create a fresh chess board in the starting position.

        :param self: The instance being initialized.
        :return: None.
        """
        self.board: chess.Board = chess.Board()

    def get_legal_moves(self) -> list[str]:
        """Return all legal moves in UCI notation for the current position.

        :param self: The board state instance.
        :return: A list of legal moves as UCI strings.
        """
        return [str(move) for move in self.board.legal_moves]

    def check_move_legality(self, move: str) -> bool:
        """Check whether a move string is legal in the current board state.

        :param move: The move in UCI format, such as "e2e4".
        :return: True when the move is legal, otherwise False.
        """
        try:
            chess_move = chess.Move.from_uci(move)
            return chess_move in self.board.legal_moves
        except ValueError:
            return False

    def make_move(self, move: str) -> bool:
        """Apply a legal move to the internal board.

        :param move: The move in UCI format to play.
        :return: True if the move was applied, otherwise False.
        """
        if self.check_move_legality(move):
            chess_move = chess.Move.from_uci(move)
            self.board.push(chess_move)
            return True
        return False
