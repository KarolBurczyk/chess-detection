class Config:
    """Runtime configuration values used by the board and piece detectors.

    :param MODEL_PATH: Path to the trained YOLO model used for piece detection.
    :param SOURCE: Camera source index or video file path.
    :param CONF: YOLO confidence threshold.
    :param IOU: Non-maximum suppression IoU threshold.
    :return: None.
    """

    MODEL_PATH = "/Users/karol/Desktop/new_chess/runs/detect/runs/chess_train/weights/best.pt"
    SOURCE = 0
    CONF = 0.1
    IOU = 0.5