from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Farm(Base, TimestampMixin):
    __tablename__ = "farms"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    latitude = Column(Float)
    longitude = Column(Float)
    region = Column(String(50))
    size_hectares = Column(Float)