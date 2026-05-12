import time
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def wait_for_db(max_retries: int = 10, delay: int = 2) -> bool:
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Подключение к БД успешно")
            return True
        except Exception as e:
            logger.warning(f"Попытка {i+1}/{max_retries}: БД недоступна ({e})")
            time.sleep(delay)
    raise RuntimeError("Не удалось подключиться к БД. Проверьте docker-compose и DATABASE_URL")