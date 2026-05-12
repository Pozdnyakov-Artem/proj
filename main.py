import sys
import time
import queue
import threading
import logging
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import json
import os
import tbot

from config import (
    SAVE_DIR, CAMERAS, DEFAULT_CAM_ID, DETECTION_THRESHOLD,
    SCALE, DEFAULT_CLASS_IDS, COCO_NAMES
)
from database import engine, SessionLocal, Base, wait_for_db
from detection.detector import Detector
from detection.zone_utils import filter_detections
from ui.zone_drawer import ZoneDrawer
from ui.camera_selector import CameraSelector
from ui.class_selector import ClassSelector
from video.stream_manager import StreamManager
from workers.save_worker import save_worker
from workers.db_worker import db_worker
import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

frame_queue = queue.Queue(maxsize=50)
db_queue = queue.Queue(maxsize=100)
stop_event = threading.Event()

SETTINGS_FILE = "camera_settings.json"


def load_camera_settings():
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            result = {}
            for cid_str, cfg in data.items():
                cid = int(cid_str)
                result[cid] = {
                    "zones": [np.array(p, dtype=np.int32) for p in cfg.get("zones", [])],
                    "classes": set(cfg.get("classes", [1]))
                }
            logger.info(f"Загружено настроек камер: {len(result)}")
            return result
    except Exception as e:
        logger.warning(f"Не удалось загрузить настройки: {e}")
        return {}


def save_camera_settings(settings: dict):
    try:
        serializable = {}
        for cid, cfg in settings.items():
            serializable[str(cid)] = {
                "zones": [p.tolist() for p in cfg["zones"]],
                "classes": list(cfg["classes"])
            }
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Настройки сохранены в {SETTINGS_FILE}")
    except Exception as e:
        logger.error(f"Ошибка сохранения настроек: {e}")

def run_camera_setup() -> int | None:
    win_name = "System Setup"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, 800, 600)

    btn = (250, 250, 300, 60)  # x, y, w, h
    clicked = False

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked
        if event == cv.EVENT_LBUTTONDOWN:
            if btn[0] <= x <= btn[0] + btn[2] and btn[1] <= y <= btn[1] + btn[3]:
                clicked = True

    cv.setMouseCallback(win_name, on_mouse)

    while not stop_event.is_set():
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        cv.putText(frame, "NO CAMERAS CONFIGURED", (150, 150),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        cv.putText(frame, "Click button to add your first source", (180, 190),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv.rectangle(frame, (btn[0], btn[1]), (btn[0]+btn[2], btn[1]+btn[3]), (40, 40, 40), -1)
        cv.rectangle(frame, (btn[0], btn[1]), (btn[0]+btn[2], btn[1]+btn[3]), (0, 200, 200), 2)
        cv.putText(frame, "+ ADD CAMERA", (btn[0] + 65, btn[1] + 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(frame, "[Q] Exit", (360, 550),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv.imshow(win_name, frame)
        key = cv.waitKey(50) & 0xFF

        if clicked:
            clicked = False
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.focus_force()
            url = simpledialog.askstring("Add Camera", "Enter RTSP/HTTP/MP4 URL:")
            root.destroy()

            if url and url.strip():
                new_id = max(CAMERAS.keys(), default=0) + 1
                CAMERAS[new_id] = url.strip()
                logger.info(f"Камера {new_id} добавлена: {url.strip()}")
                cv.destroyAllWindows()
                return new_id  # Успех → возвращаем ID

        if key in (ord('q'), 27):
            break

    cv.destroyAllWindows()
    return None  # Отмена

def main():
    wait_for_db()
    Base.metadata.create_all(bind=engine)

    detector = Detector(threshold=DETECTION_THRESHOLD)
    stream_mgr = StreamManager()
    zone_drawer = ZoneDrawer("Detection UI")

    initial_cam_id = None
    if not CAMERAS:
        initial_cam_id = run_camera_setup()
        if initial_cam_id is None:
            logger.warning("Настройка отменена. Завершение.")
            return
        target_cam_id = initial_cam_id
    else:
        target_cam_id = DEFAULT_CAM_ID

    runtime_cameras = {
        k: (v.split('/')[-1] if v else f"Cam {k}")
        for k, v in CAMERAS.items()
        if v
    }
    camera_selector = CameraSelector(
        cameras=runtime_cameras,
        current_id=DEFAULT_CAM_ID
    )

    coco_dict = {cid: name for cid, name in zip(COCO_CLASSES, COCO_NAMES)}
    class_selector = ClassSelector(coco_dict, selected_ids=DEFAULT_CLASS_IDS)

    CAMERA_SETTINGS = load_camera_settings()
    add_camera_requested = False

    def on_camera_changed(cam_id):
        old_cam_id = stream_mgr.current_id

        if old_cam_id is not None and old_cam_id != cam_id:
            CAMERA_SETTINGS[old_cam_id] = {
                "zones": [np.array(p, dtype=np.int32, copy=True) for p in zone_drawer.polygons],
                "classes": set(class_selector.selected)
            }
            logger.info(
                f"Saved Cam {old_cam_id}: {len(zone_drawer.polygons)} zones, {len(class_selector.selected)} classes")

        if cam_id not in CAMERA_SETTINGS:
            CAMERA_SETTINGS[cam_id] = {"zones": [], "classes": {1}}
            logger.info(f"New Cam {cam_id}: initialized with defaults")

        cfg = CAMERA_SETTINGS[cam_id]
        zone_drawer.polygons = [np.array(p, dtype=np.int32, copy=True) for p in cfg["zones"]]
        zone_drawer.temp_polygon = []
        class_selector.selected = set(cfg["classes"])

        if cam_id != old_cam_id:
            if stream_mgr.switch_to(cam_id):
                logger.info(f"Stream switched to Cam {cam_id}")
            else:
                logger.error(f"Failed to switch stream to Cam {cam_id}")

        logger.info(
            f"Loaded Cam {cam_id}: {len(zone_drawer.polygons)} zones, classes {sorted(class_selector.selected)}")

    def on_add_camera_request():
        nonlocal add_camera_requested
        add_camera_requested = True

    camera_selector.on_change = on_camera_changed
    camera_selector.on_add_request = on_add_camera_request

    def unified_mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if camera_selector.handle_click(x, y):
                return
            if class_selector.handle_click(x, y):
                return
        zone_drawer.mouse_callback(event, x, y, flags, param)

    t_save = threading.Thread(target=save_worker, args=(frame_queue, db_queue, stop_event, SAVE_DIR), daemon=True)
    t_db = threading.Thread(target=db_worker, args=(db_queue, stop_event, SAVE_DIR, SessionLocal), daemon=True)
    t_save.start()
    t_db.start()

    if not stream_mgr.switch_to(target_cam_id):
        logger.error(f"Не удалось открыть камеру {target_cam_id}")
        sys.exit(1)

    orig_w, orig_h = stream_mgr.get_resolution()
    fps = stream_mgr.get_fps()
    scaled_w, scaled_h = int(orig_w * SCALE), int(orig_h * SCALE)

    video_out = cv.VideoWriter('out.avi', cv.VideoWriter_fourcc(*'XVID'), fps, (scaled_w, scaled_h))

    color = sv.ColorPalette.from_hex(["#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff"])
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(scaled_w, scaled_h))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(scaled_w, scaled_h))

    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color, text_color=sv.Color.BLACK,
        text_scale=text_scale, smart_position=True
    )

    fn = 0
    paused = False
    FRAME_STEP = 1

    cv.namedWindow("Detection UI")
    cv.setMouseCallback("Detection UI", unified_mouse_callback)

    try:
        while True:
            annotated = None

            if not paused:
                ret, frame = stream_mgr.read()
                if not ret:
                    time.sleep(0.5)
                    stream_mgr.switch_to(stream_mgr.current_id)
                    continue

                fn += 1
                frame_scaled = cv.resize(frame, (scaled_w, scaled_h))
                frame_rgb = cv.cvtColor(frame_scaled, cv.COLOR_BGR2RGB)

                is_detect_frame = (fn % FRAME_STEP == 0)

                annotated = frame_scaled.copy()
                annotated = zone_drawer.draw(annotated)

                if is_detect_frame:
                    all_detections = detector.predict(frame_rgb)

                    detections = filter_detections(
                        all_detections,
                        class_ids=list(class_selector.selected),
                        polygons=zone_drawer.polygons
                    )

                    if len(detections) > 0:
                        labels = [
                            f"{coco_dict.get(cid, 'obj')} {conf:.2f}"
                            for cid, conf in zip(detections.class_id, detections.confidence)
                        ]
                        annotated = bbox_annotator.annotate(annotated, detections)
                        annotated = label_annotator.annotate(annotated, detections, labels)

                        try:
                            detection_info = None
                            if len(detections) > 0 and len(detections.confidence) > 0:
                                detection_info = {
                                    "class_id": int(detections.class_id[0]),
                                    "class_name": coco_dict.get(int(detections.class_id[0]), "unknown"),
                                    "confidence": float(detections.confidence[0]),
                                    "count": len(detections)
                                }
                                tbot.send_alert(stream_mgr.current_id,detection_info["count"],[detection_info["class_name"]])

                            frame_queue.put_nowait((
                                annotated.copy(),
                                stream_mgr.current_id,
                                time.time(),
                                detection_info
                            ))
                        except queue.Full:
                            logger.warning("Очередь сохранения переполнена")

                annotated = camera_selector.draw(annotated)
                annotated = class_selector.draw(annotated)

                status_txt = f"AI: {'ON' if is_detect_frame else 'OFF'} | Frame: {fn}"
                status_color = (0, 255, 0) if is_detect_frame else (100, 100, 100)
                cv.putText(annotated, status_txt, (10, scaled_h - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

                cv.putText(annotated, "[Q]uit [P]ause [R]eset", (10, scaled_h - 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                video_out.write(annotated)

            display_frame = annotated if annotated is not None else frame_scaled
            cv.imshow("Detection UI", display_frame)

            key = cv.waitKey(1)

            if add_camera_requested:
                add_camera_requested = False
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)

                url = simpledialog.askstring(
                    "Добавить источник видео",
                    "Введите URL (rtsp://..., http://..., file.mp4):"
                )
                root.destroy()

                if url and url.strip():
                    url = url.strip()
                    new_id = max(CAMERAS.keys(), default=0) + 1

                    CAMERAS[new_id] = url
                    runtime_cameras[new_id] = url.split('/')[-1] if "/" in url else f"File {new_id}"
                    camera_selector.cameras = runtime_cameras

                    if stream_mgr.open_stream(new_id, url):
                        if new_id not in CAMERA_SETTINGS:
                            CAMERA_SETTINGS[new_id] = {"zones": [], "classes": {1}}
                        logger.info(f"Камера {new_id} успешно добавлена")
                    else:
                        logger.error(f"Не удалось открыть камеру {new_id}")

            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('p'), ord('P')]:
                paused = not paused
            elif key == ord('r'):
                zone_drawer.clear_all()
                logger.info("Зоны очищены")

    except KeyboardInterrupt:
        logger.info("Ctrl+C — завершение")
    finally:
        if stream_mgr.current_id:
            CAMERA_SETTINGS[stream_mgr.current_id] = {
                "zones": [np.array(p, dtype=np.int32, copy=True) for p in zone_drawer.polygons],
                "classes": set(class_selector.selected)
            }

        save_camera_settings(CAMERA_SETTINGS)

        stop_event.set()
        frame_queue.join()
        db_queue.join()
        cv.destroyAllWindows()
        stream_mgr.release_all()
        video_out.release()
        logger.info("Завершено")


if __name__ == '__main__':
    main()