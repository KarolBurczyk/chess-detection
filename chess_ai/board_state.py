import chess


class BoardState:
    """Class representing the state of a chess game."""

    def __init__(self):
        """Initialize a new chess game state."""
        self.board: chess.Board = chess.Board()

    def get_legal_moves(self) -> list[str]:
        """Get a list of legal moves in UCI format.
        :return: A list of legal moves as strings in UCI format.
        """
        return list(
            str(move) for move in self.board.legal_moves
        )

    def check_move_legality(self, move: str) -> bool:
        """Check if a move is legal.

        :param move: The move to check in UCI format.
        :return: True if the move is legal, False otherwise.
        """
        try:
            chess_move = chess.Move.from_uci(move)
            return chess_move in self.board.legal_moves
        except ValueError:
            return False

    def make_move(self, move: str) -> bool:
        """Make a move on the chess board.

        :param move: The move to make in UCI format.
        :return: True if the move was successful, False otherwise.
        """
        if self.check_move_legality(move):
            chess_move = chess.Move.from_uci(move)
            self.board.push(chess_move)
            return True
        return False
