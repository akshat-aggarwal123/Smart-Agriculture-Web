from app.models import Base, TimestampMixin
from sqlalchemy import Column, Integer, String, Float, ForeignKey

class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True)
    farm_id = Column(Integer, ForeignKey("farms.id"))
    prediction_type = Column(String(50))  # 'irrigation', 'fertilizer', etc.
    input_features = Column(String)        # JSON string
    predicted_value = Column(Float)
    confidence_score = Column(Float)
    model_version = Column(String(20))