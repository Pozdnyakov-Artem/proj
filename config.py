import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

SAVE_DIR = os.getenv("SAVE_DIR")
os.makedirs(SAVE_DIR, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")

CAMERAS = {
    1: os.getenv("PATH_CAM_1"),
    2: os.getenv("PATH_CAM_2"),
    # 3: os.getenv("PATH_CAM_3", "video_file.mp4"),
}
DEFAULT_CAM_ID = int(os.getenv("DEFAULT_CAM_ID", "1"))

DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
SCALE = float(os.getenv("VIDEO_SCALE", "0.5"))
DEFAULT_CLASS_IDS = [1]

ZONE_COLOR = (0, 255, 255)
ZONE_ALPHA = 0.3
UI_BUTTON_COLOR = (70, 130, 180)
UI_BUTTON_HOVER_COLOR = (100, 149, 237)
UI_TEXT_COLOR = (255, 255, 255)

COCO_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]