from app.models import Base, TimestampMixin
from sqlalchemy import Column, Integer, Float, ForeignKey

class SoilHealth(Base, TimestampMixin):
    __tablename__ = "soil_health"
    
    id = Column(Integer, primary_key=True)
    farm_id = Column(Integer, ForeignKey("farms.id"))
    measurement_date = Column(DateTime)
    ph_level = Column(Float)
    moisture_percentage = Column(Float)
    nitrogen_level = Column(Float)
    phosphorus_level = Column(Float)
    potassium_level = Column(Float)
    sustainability_score = Column(Float)