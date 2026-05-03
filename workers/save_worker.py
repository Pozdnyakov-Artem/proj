import os
import queue
import threading
import logging
import cv2 as cv

logger = logging.getLogger(__name__)


def save_worker(frame_queue: queue.Queue, db_queue: queue.Queue,
                stop_event: threading.Event, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            if len(item) == 4:
                frame, cam_id, ts, detection_info = item
            else:
                frame, cam_id, ts = item
                detection_info = None
        except queue.Empty:
            continue

        try:
            filename = f"temp_{cam_id}_{int(ts * 1000)}.jpg"
            filepath = os.path.join(save_dir, filename)

            success, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 85])
            if success:
                with open(filepath, 'wb') as f:
                    f.write(buf.tobytes())

                db_queue.put_nowait((filepath, cam_id, detection_info))
            else:
                logger.warning(f"Ошибка кодирования кадра {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
        finally:
            frame_queue.task_done()