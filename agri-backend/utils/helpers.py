from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional

def get_db():
    """Dependency for getting database session"""
    from services.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert SQLAlchemy row to dictionary"""
    return {key: getattr(row, key) for key in row.__table__.columns.keys()}

def validate_features(features: Dict[str, Any]) -> bool:
    """Validate required features exist"""
    required_fields = ["Soil_Moisture", "Temperature", "Rainfall", "Crop_Type", "Market_Price_per_ton"]
    return all(field in features for field in required_fields)