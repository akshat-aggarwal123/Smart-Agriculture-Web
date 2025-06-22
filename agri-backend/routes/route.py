from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.ml_service import MLService
from utils.helpers import get_db
from models.farm import Farm

router = APIRouter()

def get_ml_service():
    return MLService()

@router.get("/farms")
async def get_farms(db: Session = Depends(get_db)):
    return db.query(Farm).all()

@router.get("/farms/{farm_id}")
async def get_farm(farm_id: int, db: Session = Depends(get_db)):
    return db.query(Farm).filter(Farm.id == farm_id).first()

@router.post("/predict/yield")
async def predict_yield(farm_id: int, db: Session = Depends(get_db), ml_service: MLService = Depends(get_ml_service)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    
    # Dummy features (replace with real data from Redis or database)
    features = {
        "soil_moisture": 45.2,
        "temperature": 28.5,
        "rainfall": 120.0,
        "crop_type": "Wheat",
        "market_price": 250.0
    }
    
    prediction = ml_service.predict_yield(features)
    return prediction