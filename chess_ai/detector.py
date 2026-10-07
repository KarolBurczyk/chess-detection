from __future__ import annotations

import cv2
import chess

from config import Config
from board_detector import BoardDetector
from piece_detector import PieceDetector


class DetectorApp:
    def __init__(self):
        self.board = BoardDetector()
        self.pieces = PieceDetector()
        self.cap = cv2.VideoCapture(Config.SOURCE)

    def run(self):
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

            cv2.imshow("Szachy YOLO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("r"):
                self.board = BoardDetector()

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    app = DetectorApp()
    app.run()


if __name__ == "__main__":
    main()
