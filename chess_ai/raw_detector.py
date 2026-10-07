"""Quick YOLO-and-board prototype used for debugging and simple live detection.

This module is more lightweight than the main application loop and is useful for
checking whether detections and board geometry align before running the full game
tracker pipeline.
"""

from __future__ import annotations

import cv2
import chess

from config import Config
from board_detector import BoardDetector
from piece_detector import PieceDetector


class DetectorApp:
    """Run a lightweight detection loop for debugging the camera and board geometry."""

    def __init__(self) -> None:
        """Create the detector app and initialize camera, board detection, and YOLO detection.

        :return: None.
        """
        self.board = BoardDetector()
        self.pieces = PieceDetector()
        self.cap = cv2.VideoCapture(Config.SOURCE)

    def run(self) -> None:
        """Process camera frames until the user quits.

        :return: None.
        """
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = self.board.update(frame)
            detections = self.pieces.detect(frame)

            for detection in detections:
                square = None
                if self.board.H is not None:
                    square = self.board.to_square(
                        (detection.x1 + detection.x2) / 2,
                        detection.y2 - 0.15 * (detection.y2 - detection.y1),
                        self.board.H,
                    )

                label = detection.label
                if square is not None:
                    label = f"{chess.square_name(square)} {label}"

                cv2.rectangle(
                    frame,
                    (detection.x1, detection.y1),
                    (detection.x2, detection.y2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{label} {detection.conf:.2f}",
                    (detection.x1, max(detection.y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            cv2.imshow("Chess YOLO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("r"):
                self.board = BoardDetector()

        self.cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Run the quick detector app.

    :return: None.
    """
    app = DetectorApp()
    app.run()


if __name__ == "__main__":
    main()
