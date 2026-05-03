import os
import queue
import threading
import logging
from models import IMG

logger = logging.getLogger(__name__)


def db_worker(db_queue: queue.Queue, stop_event: threading.Event,
              save_dir: str, SessionLocal):
    while not stop_event.is_set():
        try:
            item = db_queue.get(timeout=0.5)
            if len(item) == 3:
                filepath, cam_id, detection_info = item
            else:
                filepath, cam_id = item
                detection_info = None
        except queue.Empty:
            continue

        try:
            with SessionLocal() as session:
                new_img = IMG(
                    cam_id=cam_id,
                    img_path=filepath,
                    detected_class=detection_info["class_name"] if detection_info else None,
                    confidence=detection_info["confidence"] if detection_info else None
                )
                session.add(new_img)
                session.commit()
                db_id = new_img.id

                ext = os.path.splitext(filepath)[1]
                new_filename = f"cam{cam_id}_{db_id}{ext}"
                new_path = os.path.join(save_dir, new_filename)
                os.replace(filepath, new_path)

                new_img.img_path = new_path
                session.commit()
                logger.debug(f"Сохранено: {new_path} | class={new_img.detected_class}, conf={new_img.confidence}")

        except Exception as e:
            logger.error(f"Ошибка БД: {e}")
            session.rollback()
        finally:
            db_queue.task_done()