import cv2 as cv
import numpy as np
from typing import Dict, List, Callable, Optional

class ClassSelector:
    PANEL_WIDTH = 250
    ITEM_HEIGHT = 28
    MAX_VISIBLE = 12
    MARGIN = 15
    FONT = cv.FONT_HERSHEY_SIMPLEX

    def __init__(self, available_classes: Dict[int, str], selected_ids: List[int] = None):
        self.available = available_classes
        self.class_ids = sorted(available_classes.keys())
        self.selected = set(selected_ids if selected_ids is not None else [1])
        self.visible = False
        self.scroll_offset = 0
        self.on_change: Optional[Callable[[List[int]], None]] = None
        self.panel_rect = [0, 0, 0, 0]

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.scroll_offset = 0

    def _get_visible_items(self):
        return self.class_ids[self.scroll_offset : self.scroll_offset + self.MAX_VISIBLE]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        x, y = self.MARGIN, self.MARGIN
        btn_w, btn_h = 130, 32

        if self.visible:
            bg_color = (0, 160, 100)
            border_color = (255, 255, 0)
            btn_text = f"Classes ({len(self.selected)})"
        else:
            bg_color = (70, 130, 180)
            border_color = (255, 255, 255)
            btn_text = f"Classes ({len(self.selected)})"

        cv.rectangle(frame, (x, y), (x + btn_w, y + btn_h), bg_color, -1)
        cv.rectangle(frame, (x, y), (x + btn_w, y + btn_h), border_color, 2 if self.visible else 1)

        text_w, text_h = cv.getTextSize(btn_text, self.FONT, 0.6, 1)[0]
        tx = x + (btn_w - text_w) // 2
        ty = y + (btn_h + text_h) // 2
        cv.putText(frame, btn_text, (tx, ty), self.FONT, 0.6, (255, 255, 255), 1)

        if not self.visible:
            return frame

        panel_gap = 5
        self.panel_h = (self.MAX_VISIBLE + 2) * self.ITEM_HEIGHT + 35
        self.panel_rect = [x, y + btn_h + panel_gap, self.PANEL_WIDTH, self.panel_h]
        px, py, pw, ph = self.panel_rect

        cv.rectangle(frame, (px, py), (px + pw, py + ph), (25, 25, 25), -1)
        cv.rectangle(frame, (px, py), (px + pw, py + ph), (120, 120, 120), 1)

        cv.putText(frame, f"Selected: {len(self.selected)}/{len(self.available)}",
                   (px + 10, py + 18), self.FONT, 0.5, (200, 200, 200), 1)

        btn_y = py + 28
        cv.rectangle(frame, (px, btn_y), (px + pw, btn_y + self.ITEM_HEIGHT), (50, 50, 50), -1)
        cv.putText(frame, "Prev", (px + 10, btn_y + 19), self.FONT, 0.5, (200, 200, 200), 1)

        down_y = py + 28 + (self.MAX_VISIBLE + 1) * self.ITEM_HEIGHT
        cv.rectangle(frame, (px, down_y), (px + pw, down_y + self.ITEM_HEIGHT), (50, 50, 50), -1)
        cv.putText(frame, "Next", (px + 10, down_y + 19), self.FONT, 0.5, (200, 200, 200), 1)

        for i, cid in enumerate(self._get_visible_items()):
            iy = py + 28 + (i + 1) * self.ITEM_HEIGHT
            name = self.available[cid]

            bx1, by1 = px + 10, iy + 4
            bx2, by2 = bx1 + 16, by1 + 16
            color = (0, 200, 0) if cid in self.selected else (80, 80, 80)
            cv.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
            if cid in self.selected:
                cv.line(frame, (bx1 + 3, by1 + 8), (bx1 + 7, by1 + 12), (255, 255, 255), 2)
                cv.line(frame, (bx1 + 7, by1 + 12), (bx1 + 13, by1 + 6), (255, 255, 255), 2)

            cv.putText(frame, f"{cid:3d}: {name}", (bx2 + 8, iy + 18),
                       self.FONT, 0.45, (255, 255, 255), 1)
        return frame

    def handle_click(self, x: int, y: int) -> bool:
        btn_x1, btn_y1 = self.MARGIN, self.MARGIN
        btn_x2, btn_y2 = btn_x1 + 130, btn_y1 + 32

        if not self.visible:
            if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                self.toggle()
                return True
            return False

        px, py, pw, ph = self.panel_rect

        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            self.toggle()
            return True

        if not (px <= x <= px + pw and py <= y <= py + ph):
            self.visible = False
            return True

        if px <= x <= px + pw and py + 28 <= y <= py + 28 + self.ITEM_HEIGHT:
            self.scroll_offset = max(0, self.scroll_offset - 1)
            return True
        down_y = py + 28 + (self.MAX_VISIBLE + 1) * self.ITEM_HEIGHT
        if px <= x <= px + pw and down_y <= y <= down_y + self.ITEM_HEIGHT:
            max_off = max(0, len(self.class_ids) - self.MAX_VISIBLE)
            self.scroll_offset = min(max_off, self.scroll_offset + 1)
            return True

        for i, cid in enumerate(self._get_visible_items()):
            iy = py + 28 + (i + 1) * self.ITEM_HEIGHT
            if px + 10 <= x <= px + 26 and iy + 4 <= y <= iy + 20:
                if cid in self.selected:
                    self.selected.remove(cid)
                else:
                    self.selected.add(cid)
                if self.on_change:
                    self.on_change(list(self.selected))
                return True

        return False