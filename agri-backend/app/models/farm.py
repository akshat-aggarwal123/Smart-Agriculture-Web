from app.models import Base, TimestampMixin
from sqlalchemy import Column, Integer, String, Float

class Farm(Base, TimestampMixin):
    __tablename__ = "farms"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    latitude = Column(Float)
    longitude = Column(Float)
    region = Column(String(50))
    size_hectares = Column(Float)