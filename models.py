from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class IMG(Base):
    __tablename__ = "frame_with_object"

    id = Column(Integer, primary_key=True, index=True)
    cam_id = Column(Integer, index=True)
    img_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)