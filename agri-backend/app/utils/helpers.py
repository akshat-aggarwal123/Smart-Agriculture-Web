from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional

def get_db():
    """Dependency for getting database session"""
    from app.services.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert SQLAlchemy row to dictionary"""
    return {key: getattr(row, key) for key in row.__table__.columns.keys()}

def rows_to_dict(rows: List[Any]) -> List[Dict[str, Any]]:
    """Convert list of SQLAlchemy rows to list of dictionaries"""
    if not rows:
        return []
    return [row_to_dict(row) for row in rows]

def get_farm_by_id(db: Session, farm_id: int) -> Optional[Dict]:
    """Get farm by ID (example helper function)"""
    from app.models.farm import Farm
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    return row_to_dict(farm) if farm else None