import chess
import cv2
import numpy as np

from board_detector import BoardDetector
from board_state import BoardState
from board_view import BoardView
from config import Config
from move_tracker import MoveTracker
from piece_detector import PieceDetector


def draw_status(frame: np.ndarray, state: BoardState, tracker: MoveTracker) -> None:
    """Draw the current game status on the camera frame.

    :param frame: OpenCV frame that will be annotated.
    :param state: Current chess state used to decide turn and end-of-game info.
    :param tracker: Tracker object that keeps candidate move and match statistics.
    :return: None.
    """
    side = "biale" if state.board.turn is True else "czarne"
    hits, total = tracker.match
    lines = [
        f"Tura: {side} | ruchow: {len(tracker.history)} | zgodnosc: {hits}/{total}",
    ]
    if tracker.history:
        lines.append(f"Ostatni ruch: {tracker.history[-1]}")
    if tracker.candidate:
        lines.append(f"Kandydat: {tracker.candidate} ({tracker.streak}/{tracker.confirm_frames})")
    if state.board.is_game_over():
        lines.append(f"Koniec partii: {state.board.result()}")

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 55 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def main() -> None:
    """Run the live chess-board detector and tracker loop.

    :param None: No arguments are required for this entrypoint.
    :return: None.
    """
    cap = cv2.VideoCapture(Config.SOURCE)
    board_detector = BoardDetector()
    piece_detector = PieceDetector(board_detector=board_detector)
    state = BoardState()
    tracker = MoveTracker(state, board_detector, piece_detector.class_names)
    view = BoardView()

    calibration_frames: list[np.ndarray] = []
    while len(calibration_frames) < 12:
        ok, frame = cap.read()
        if not ok:
            break
        frame = board_detector.update(frame)
        if board_detector.H is not None:
            calibration_frames.append(frame.copy())
        if len(calibration_frames) >= 12:
            break

    if calibration_frames:
        expected_start = {
            square: ("white" if piece.color == chess.WHITE else "black")
            for square, piece in state.board.piece_map().items()
        }
        piece_detector.auto_tune(calibration_frames, expected=expected_start)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = board_detector.update(frame)
        detections = piece_detector.detect(frame)

        if board_detector.H is not None:
            san = tracker.update(detections)
            if san:
                ply = len(tracker.history)
                prefix = f"{(ply + 1) // 2}." if ply % 2 == 1 else f"{ply // 2}..."
                print(f"{prefix} {san}")
            draw_status(frame, state, tracker)

        piece_detector.draw(frame, detections)
        cv2.imshow("Szachy", frame)
        cv2.imshow("Plansza", view.render(state.board, tracker.history, tracker.candidate))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            board_detector.reset()
        elif key == ord("n"):
            tracker.reset()
        elif key == ord("f"):
            view.toggle_flip()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()