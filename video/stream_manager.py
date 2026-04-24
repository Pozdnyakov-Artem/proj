import cv2 as cv
import numpy as np
import logging
from typing import Dict, Optional
from config import CAMERAS

logger = logging.getLogger(__name__)


class StreamManager:
    def __init__(self):
        self.streams: Dict[int, cv.VideoCapture] = {}
        self.current_id: Optional[int] = None

    def open_stream(self, cam_id: int, url: str) -> bool:
        if cam_id in self.streams:
            return self.streams[cam_id].isOpened()

        cap = cv.VideoCapture(url)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть камеру {cam_id}: {url}")
            return False

        if url.startswith(('rtsp://', 'rtsps://')):
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.streams[cam_id] = cap
        logger.info(f"Камера {cam_id} открыта")
        return True

    def switch_to(self, cam_id: int) -> bool:
        if cam_id not in CAMERAS:
            logger.error(f"Камера {cam_id} не найдена в конфиге")
            return False

        if not self.open_stream(cam_id, CAMERAS[cam_id]):
            return False

        self.current_id = cam_id
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.current_id is None or self.current_id not in self.streams:
            return False, None

        cap = self.streams[self.current_id]
        return cap.read()

    def get_fps(self) -> float:
        if self.current_id and self.current_id in self.streams:
            return self.streams[self.current_id].get(cv.CAP_PROP_FPS)
        return 30.0

    def get_resolution(self) -> tuple[int, int]:
        if self.current_id and self.current_id in self.streams:
            cap = self.streams[self.current_id]
            w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 1280, 720

    def release_all(self):
        for cam_id, cap in self.streams.items():
            cap.release()
            logger.info(f"Камера {cam_id} закрыта")
        self.streams.clear()