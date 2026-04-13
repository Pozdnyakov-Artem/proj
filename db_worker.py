import os
import queue
import threading
import logging

from models import IMG

logger = logging.getLogger(__name__)

def db_worker(db_queue: queue.Queue, stop_event: threading.Event, save_dir: str, SessionLocal):
    while not stop_event.is_set():
        try:
            filepath, cam_id = db_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            with SessionLocal() as session:

                new_img = IMG(cam_id=cam_id, img_path=filepath)
                session.add(new_img)
                session.commit()
                db_id = new_img.id

                new_filename = f"cam{cam_id}_{db_id}.jpg"
                new_path = os.path.join(save_dir, new_filename)
                os.replace(filepath, new_path)

                new_img.img_path = new_path
                session.commit()
        except Exception as e:
            logger.error(f"❌ Ошибка БД: {e}")
        finally:
            db_queue.task_done()
