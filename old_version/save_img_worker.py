import logging
import cv2
import threading
import queue
import os

logger = logging.getLogger(__name__)

def save_worker(frame_queue: queue.Queue, db_queue: queue.Queue,
                stop_event: threading.Event, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    while not stop_event.is_set():
        try:
            frame, cam_id, ts = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            filename = f"temp_{int(ts*1000)}.jpg"
            filepath = os.path.join(save_dir, filename)
            success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                with open(filepath, 'wb') as f:
                    f.write(buf.tobytes())
                db_queue.put_nowait((filepath, cam_id))
            else:
                logger.warning(f"Ошибка кодирования кадра {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {str(e)}")
        finally:
            frame_queue.task_done()