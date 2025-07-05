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
    farm_name = Column(String(100))
    soil_moisture = Column("Soil_Moisture", Float)  # Must match training feature names
    temperature = Column("Temperature", Float)
    rainfall = Column("Rainfall", Float)
    crop_type = Column("Crop_Type", String(50))
    market_price = Column("Market_Price_per_ton", Float)