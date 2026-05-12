# ui/camera_selector.py
import cv2 as cv
import numpy as np
from typing import Dict, Optional, Callable


class CameraSelector:
    BUTTON_WIDTH = 140
    BUTTON_HEIGHT = 30
    ADD_BTN_SIZE = 30
    MARGIN = 15
    FONT = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6

    def __init__(self, cameras: Dict[int, str], current_id: int):
        self.cameras = cameras
        self.current_id = current_id
        self.is_open = False
        self.on_change: Optional[Callable[[int], None]] = None
        self.on_add_request: Optional[Callable[[], None]] = None
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.add_btn = (0, 0, 0, 0)

    def _update_coords(self, frame_w: int, frame_h: int):
        self.x2 = frame_w - self.MARGIN
        self.y2 = self.MARGIN + self.BUTTON_HEIGHT
        self.x1 = self.x2 - self.BUTTON_WIDTH
        self.y1 = self.y2 - self.BUTTON_HEIGHT

        ax2 = self.x1 - self.MARGIN
        ax1 = ax2 - self.ADD_BTN_SIZE
        self.add_btn = (ax1, self.y1, ax2, self.y2)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        self._update_coords(frame.shape[1], frame.shape[0])

        cv.rectangle(frame, (self.add_btn[0], self.add_btn[1]),
                     (self.add_btn[2], self.add_btn[3]), (40, 160, 40), -1)
        cv.putText(frame, "+", (self.add_btn[0] + 7, self.add_btn[3] - 8),
                   self.FONT, 0.8, (255, 255, 255), 2)

        color = (100, 149, 237) if self.is_open else (70, 130, 180)
        cv.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), color, -1)
        cv.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 255), 1)

        text = f"Cam {self.current_id}"
        text_w, text_h = cv.getTextSize(text, self.FONT, self.FONT_SCALE, 1)[0]
        tx = self.x1 + (self.BUTTON_WIDTH - text_w) // 2
        ty = self.y1 + (self.BUTTON_HEIGHT + text_h) // 2
        cv.putText(frame, text, (tx, ty), self.FONT, self.FONT_SCALE, (255, 255, 255), 1)

        if self.is_open:
            self._draw_dropdown(frame)
        return frame

    def _draw_dropdown(self, frame: np.ndarray):
        for i, cam_id in enumerate(self.cameras.keys()):
            y1 = self.y2 + i * 28
            y2 = y1 + 28
            color = (100, 149, 237) if cam_id == self.current_id else (50, 50, 50)
            cv.rectangle(frame, (self.x1, y1), (self.x2, y2), color, -1)
            cv.rectangle(frame, (self.x1, y1), (self.x2, y2), (150, 150, 150), 1)

            txt = f"{cam_id}: {self.cameras[cam_id][:18]}"
            tw, th = cv.getTextSize(txt, self.FONT, 0.5, 1)[0]
            cv.putText(frame, txt, (self.x1 + 8, y1 + (28 + th) // 2), self.FONT, 0.5, (255, 255, 255), 1)

    def handle_click(self, x: int, y: int) -> bool:
        ax1, ay1, ax2, ay2 = self.add_btn
        if ax1 <= x <= ax2 and ay1 <= y <= ay2:
            if self.on_add_request:
                self.on_add_request()
            return True

        if self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2:
            self.is_open = not self.is_open
            return True

        if self.is_open:
            for i, cam_id in enumerate(self.cameras.keys()):
                item_y1 = self.y2 + i * 28
                item_y2 = item_y1 + 28
                if self.x1 <= x <= self.x2 and item_y1 <= y <= item_y2:
                    self.is_open = False
                    if cam_id != self.current_id:
                        self.current_id = cam_id
                        if self.on_change:
                            self.on_change(cam_id)
                    return True

        return False