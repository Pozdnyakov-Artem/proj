from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, func, Float
from database import Base


class IMG(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    cam_id = Column(Integer, nullable=False)
    img_path = Column(String, nullable=False)
    detected_class = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    zone_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    url = Column(String, nullable=False)
    is_active = Column(Integer, default=1)