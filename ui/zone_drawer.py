import cv2 as cv
import numpy as np
from typing import List, Callable


class ZoneDrawer:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.temp_polygon: List[List[int]] = []
        self.polygons: List[np.ndarray] = []
        self.on_update: Callable = None

    def set_callback(self, callback: Callable):
        self.on_update = callback

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.temp_polygon.append([x, y])
        elif event == cv.EVENT_RBUTTONDOWN and len(self.temp_polygon) > 1:
            poly = np.array(self.temp_polygon, dtype=np.int32).reshape(-1, 2)
            self.polygons.append(poly)
            self.temp_polygon = []
            if self.on_update:
                self.on_update(self.polygons)

    def draw(self, frame: np.ndarray, color=(0, 255, 255), alpha=0.3):
        overlay = frame.copy()
        if self.polygons:
            cv.fillPoly(overlay, self.polygons, color)
            cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv.polylines(frame, self.polygons, True, color, 2)
        if len(self.temp_polygon) > 1:
            cv.polylines(frame, [np.array(self.temp_polygon)], True, (255, 255, 0), 1)
        return frame

    def clear_temp(self):
        self.temp_polygon = []

    def clear_all(self):
        self.polygons = []
        self.temp_polygon = []
        if self.on_update:
            self.on_update(self.polygons)