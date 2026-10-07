from __future__ import annotations

import cv2
import numpy as np
import chess


class BoardDetector:
    """Detect the chessboard and estimate the 8x8 homography from a camera frame.

    The key change from the previous version is that board rectangle candidates are
    not selected by the largest area alone; instead, each candidate is scored by how
    well it matches the chessboard pattern of alternating light and dark squares.
    """

    def __init__(
        self,
        warp: int = 800,
        history_frames: int = 30,
        max_margin: float = 0.10,
        bottom_left: str = "h1",
        min_fit: float = 0.30,
    ) -> None:
        """Create a board detector with calibration and detection parameters.

        :param warp: Size of the synthetic top-down board image used for calibration.
        :param history_frames: Number of frames kept to stabilize the detected corners.
        :param max_margin: Maximum margin around the board that may be cropped away.
        :param bottom_left: Square used as the board origin. For example, "h1" is the bottom-left square.
        :param min_fit: Minimum checkerboard-fit score required to accept a candidate.
        :return: None.
        """
        self.WARP = warp
        self.CELL = self.WARP // 8
        self.HISTORY_FRAMES = history_frames
        self.MAX_MARGIN = max_margin
        self.BOTTOM_LEFT = bottom_left
        self.MIN_FIT = min_fit
        self.SCORE_SIZE = 400
        self.ROT_K = self._find_rotation()
        self.message = ""
        self.last_candidate_score = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset the internal calibration state and start from a fresh board detection attempt.

        :return: None.
        """
        self.corners_history: list[np.ndarray] = []
        self.gray_frames: list[np.ndarray] = []
        self.board_corners: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.Hinv: np.ndarray | None = None
        self.margins: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self.fit_score = 0.0

    # ----------------------------------------------------------------------
    # Orientation
    # ----------------------------------------------------------------------
    @staticmethod
    def _rotate_cell(col: int, row: int) -> tuple[int, int]:
        """Rotate a chessboard cell index by 90 degrees.

        :param col: Column index in the current coordinate system.
        :param row: Row index in the current coordinate system.
        :return: Rotated column and row pair.
        """
        return 7 - row, col

    def _find_rotation(self) -> int:
        """Find the board rotation offset needed to map image coordinates to chess coordinates.

        :return: Rotation count in 0..3.
        """
        target = chess.parse_square(self.BOTTOM_LEFT)
        for k in range(4):
            c, r = 0, 7
            for _ in range(k):
                c, r = self._rotate_cell(c, r)
            if chess.square(c, 7 - r) == target:
                return k
        raise ValueError(f"BOTTOM_LEFT must be one of the board corners, got: {self.BOTTOM_LEFT}")

    # ----------------------------------------------------------------------
    # Board rectangle candidates
    # ----------------------------------------------------------------------
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order a quadrilateral as top-left, top-right, bottom-right, bottom-left.

        Sorting by angle around the center works even when the board is rotated in the image,
        which avoids corner duplication around a ~45 degree rotation.

        :param pts: Four source points describing a quadrilateral.
        :return: Ordered corner array in the expected chessboard layout.
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(ang)]
        start = int(np.argmin(pts.sum(axis=1)))
        return np.roll(pts, -start, axis=0).astype(np.float32)

    @staticmethod
    def board_candidate_is_valid(corners: np.ndarray, frame_shape) -> bool:
        """Check whether a quadrilateral is a plausible chessboard candidate.

        :param corners: Candidate rectangle corners in image space.
        :param frame_shape: Frame shape in the form (height, width, channels).
        :return: True when the rectangle looks like a valid board boundary.
        """
        if corners is None or corners.shape != (4, 2):
            return False
        h, w = frame_shape[:2]
        area = abs(cv2.contourArea(corners.astype(np.float32)))
        frame_area = float(w * h)
        if area < 0.05 * frame_area or area > 0.90 * frame_area:
            return False

        if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
            return False

        side_lengths = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        if min(side_lengths) <= 0:
            return False
        if max(side_lengths) / min(side_lengths) > 4.0:
            return False

        center = corners.mean(axis=0)
        if not (w * 0.05 <= center[0] <= w * 0.95 and h * 0.05 <= center[1] <= h * 0.95):
            return False
        return True

    @staticmethod
    def _binary_maps(gray: np.ndarray) -> list[np.ndarray]:
        """Create multiple binary maps used to extract board-candidate contours.

        :param gray: Grayscale frame.
        :return: List of edge and thresholded binary maps.
        """
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        med = float(np.median(blur))
        lo = int(max(0.0, 0.66 * med))
        hi = int(min(255.0, max(1.33 * med, lo + 20)))

        maps: list[np.ndarray] = []
        e1 = cv2.Canny(blur, lo, hi)
        maps.append(cv2.morphologyEx(e1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))
        e2 = cv2.Canny(blur, 30, 90)
        maps.append(cv2.morphologyEx(e2, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blob = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        maps.append(blob)
        maps.append(cv2.bitwise_not(blob))
        return maps

    def _candidates(self, frame_shape, gray: np.ndarray) -> list[np.ndarray]:
        """Collect board candidate quadrilaterals from the current frame.

        :param frame_shape: Image shape in (height, width, channels) format.
        :param gray: Grayscale image.
        :return: List of candidate board corners.
        """
        frame_area = frame_shape[0] * frame_shape[1]
        quads: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()

        for bmap in self._binary_maps(gray):
            contours, _ = cv2.findContours(bmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            hulls = []
            for c in contours:
                _, _, bw, bh = cv2.boundingRect(c)
                if bw * bh < 0.05 * frame_area:
                    continue
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)
                if 0.05 * frame_area <= area <= 0.95 * frame_area:
                    hulls.append((area, hull))
            hulls.sort(key=lambda t: -t[0])

            for _, hull in hulls[:8]:
                peri = cv2.arcLength(hull, True)
                for eps in (0.01, 0.02, 0.03, 0.05):
                    approx = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(approx) == 4:
                        q = self.order_points(approx.reshape(4, 2))
                        if self.board_candidate_is_valid(q, frame_shape):
                            key = tuple(np.round(q / 20.0).astype(int).ravel())
                            if key not in seen:
                                seen.add(key)
                                quads.append(q)
                        break
        return quads

    def _score_quad(self, quad: np.ndarray, gray: np.ndarray) -> float:
        """Score how strongly a quadrilateral resembles a checkerboard pattern.

        :param quad: Candidate board rectangle corners.
        :param gray: Current grayscale frame.
        :return: Checkerboard fit score in the range [0, 1].
        """
        S = self.SCORE_SIZE
        dst = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
        H0 = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(gray, H0, (S, S))
        integ = cv2.integral(warped, sdepth=cv2.CV_64F)
        return max(
            self.checker_score(integ, m, m, m, m)
            for m in np.arange(0.0, self.MAX_MARGIN + 1e-9, 0.01)
        )

    def detect_board_corners(self, frame) -> np.ndarray | None:
        """Find the best board rectangle candidate in the current frame.

        :param frame: Input camera frame.
        :return: The detected board corners or None when the board is not detected well enough.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best, best_score = None, 0.0
        for q in self._candidates(frame.shape, gray):
            s = self._score_quad(q, gray)
            if s > best_score:
                best, best_score = q, s
        self.last_candidate_score = best_score
        if best_score < self.MIN_FIT:
            return None
        return best

    # ----------------------------------------------------------------------
    # Checkerboard grid fit (excluding the border with board labels)
    # ----------------------------------------------------------------------
    def checker_score(self, integ: np.ndarray, left: float, top: float, right: float, bottom: float) -> float:
        """Estimate how closely the region matches an 8x8 alternating chessboard pattern.

        :param integ: Integral image for the warped board.
        :param left: Left crop fraction.
        :param top: Top crop fraction.
        :param right: Right crop fraction.
        :param bottom: Bottom crop fraction.
        :return: Score in the range [0, 1], where higher is more board-like.
        """
        S = integ.shape[0] - 1
        xi = np.round(np.linspace(left * S, (1 - right) * S, 9)).astype(int)
        yi = np.round(np.linspace(top * S, (1 - bottom) * S, 9)).astype(int)
        X0, Y0 = np.meshgrid(xi[:-1], yi[:-1])
        X1, Y1 = np.meshgrid(xi[1:], yi[1:])
        w, h = X1 - X0, Y1 - Y0
        ix, iy = (w * 0.2).astype(int), (h * 0.2).astype(int)

        def rect_sum(x0, y0, x1, y1):
            return integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]

        full = rect_sum(X0, Y0, X1, Y1)
        center = rect_sum(X0 + ix, Y0 + iy, X1 - ix, Y1 - iy)
        area_ring = np.maximum(w * h - (w - 2 * ix) * (h - 2 * iy), 1)
        ring = (full - center) / area_ring

        pattern = (np.indices((8, 8)).sum(axis=0) % 2) * 2 - 1
        v = ring - ring.mean()
        denom = np.linalg.norm(v) * np.linalg.norm(pattern)
        if denom < 1e-6:
            return 0.0
        return abs(float((v * pattern).sum())) / denom

    def fit_margins(self, warped_gray: np.ndarray):
        """Tune the crop margins to maximize the checkerboard fit score.

        :param warped_gray: Top-down board image with the current board geometry.
        :return: A tuple containing the best margins and the final score.
        """
        integ = cv2.integral(warped_gray, sdepth=cv2.CV_64F)

        best_m = [0.0] * 4
        best = -1.0
        for m in np.arange(0.0, self.MAX_MARGIN + 1e-9, 0.005):
            score = self.checker_score(integ, m, m, m, m)
            if score > best:
                best = score
                best_m = [float(m)] * 4

        for step in (0.01, 0.005, 0.0025, 0.00125):
            improved = True
            while improved:
                improved = False
                for i in range(4):
                    for d in (-step, step):
                        cand = best_m.copy()
                        cand[i] = float(np.clip(cand[i] + d, 0.0, self.MAX_MARGIN))
                        score = self.checker_score(integ, *cand)
                        if score > best + 1e-6:
                            best = score
                            best_m = cand
                            improved = True
        return best_m, best

    # ----------------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------------
    def calibrate(self, corners: np.ndarray, gray_frames: list[np.ndarray]):
        """Compute the homography and crop transform that map the detected board to the 8x8 plane.

        :param corners: Detected board corners in the image plane.
        :param gray_frames: Previous grayscale frames used for stable board calibration.
        :return: A tuple with the homography, inverse homography, crop margins, and fit score.
        """
        dst = np.float32([[0, 0], [self.WARP, 0], [self.WARP, self.WARP], [0, self.WARP]])
        H0 = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)

        warped = [cv2.warpPerspective(g, H0, (self.WARP, self.WARP)) for g in gray_frames[::2]]
        med = np.median(np.stack(warped), axis=0).astype(np.uint8)
        med = cv2.GaussianBlur(med, (5, 5), 0)

        margins, score = self.fit_margins(med)
        left, top, right, bottom = margins
        x0, y0, x1, y1 = left * self.WARP, top * self.WARP, (1 - right) * self.WARP, (1 - bottom) * self.WARP
        sx, sy = self.WARP / (x1 - x0), self.WARP / (y1 - y0)
        M = np.float64([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])
        H = M @ H0
        return H, np.linalg.inv(H), (left, top, right, bottom), score

    def to_square(self, px: float, py: float, H: np.ndarray | None = None):
        """Transform a pixel point into a chessboard square index.

        :param px: Point x coordinate in image space.
        :param py: Point y coordinate in image space.
        :param H: Optional homography to use instead of the current calibration.
        :return: The mapped chess square or None if the point falls outside the board.
        """
        H = self.H if H is None else H
        if H is None:
            return None
        p = cv2.perspectiveTransform(np.float32([[[px, py]]]), H)[0][0]
        col, row = int(p[0] // self.CELL), int(p[1] // self.CELL)
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        for _ in range(self.ROT_K):
            col, row = self._rotate_cell(col, row)
        return chess.square(col, 7 - row)

    def warp_to_frame(self, points, Hinv):
        """Project points from the warped board space back to the camera frame.

        :param points: Points in board space.
        :param Hinv: Inverse homography.
        :return: Projected points in the original image coordinate system.
        """
        pts = np.float32(points).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)

    def warped_view(self, frame) -> np.ndarray | None:
        """Return the top-down board view used for debugging or inspection.

        :param frame: Input camera frame.
        :return: The rectified board view or None if the board is not calibrated yet.
        """
        if self.H is None:
            return None
        return cv2.warpPerspective(frame, self.H, (self.WARP, self.WARP))

    def draw_grid(self, frame, Hinv):
        """Overlay the board grid lines onto the source frame.

        :param frame: Source image to draw on.
        :param Hinv: Inverse homography used to project the grid back to the image.
        :return: None.
        """
        for i in range(9):
            a, b = self.warp_to_frame([[i * self.CELL, 0], [i * self.CELL, self.WARP]], Hinv).astype(int)
            cv2.line(frame, tuple(a), tuple(b), (255, 200, 0), 1)
            a, b = self.warp_to_frame([[0, i * self.CELL], [self.WARP, i * self.CELL]], Hinv).astype(int)
            cv2.line(frame, tuple(a), tuple(b), (255, 200, 0), 1)

        bl = self.warp_to_frame([[self.CELL / 2, self.WARP - self.CELL / 2]], Hinv)[0].astype(int)
        cv2.putText(frame, self.BOTTOM_LEFT, tuple(bl), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ----------------------------------------------------------------------
    # Called on every frame
    # ----------------------------------------------------------------------
    def update(self, frame):
        """Track the board in the current frame and draw calibration information on it.

        :param frame: Input camera image.
        :return: The image with the board overlay drawn on top.
        """
        if self.board_corners is None:
            corners = self.detect_board_corners(frame)
            if corners is not None:
                self.corners_history.append(corners)
                self.gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 1)

            if len(self.corners_history) >= self.HISTORY_FRAMES:
                board_corners = np.median(np.stack(self.corners_history), axis=0).astype(np.float32)
                H, Hinv, margins, score = self.calibrate(board_corners, self.gray_frames)
                if score >= self.MIN_FIT:
                    self.board_corners, self.H, self.Hinv = board_corners, H, Hinv
                    self.margins, self.fit_score = margins, score
                    self.message = ""
                else:
                    self.reset()
                    self.message = f"Weak fit ({score:.2f}), searching again"

            cv2.putText(
                frame,
                f"Looking for board... {len(self.corners_history)}/{self.HISTORY_FRAMES} "
                f"(best candidate {self.last_candidate_score:.2f})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            if self.message:
                cv2.putText(frame, self.message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        cv2.polylines(frame, [self.board_corners.astype(int)], True, (255, 0, 0), 2)
        if self.Hinv is not None:
            self.draw_grid(frame, self.Hinv)
        left, top, right, bottom = self.margins
        cv2.putText(
            frame,
            f"board crop L{left:.3f} T{top:.3f} R{right:.3f} B{bottom:.3f} | fit {self.fit_score:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
        )
        return frame