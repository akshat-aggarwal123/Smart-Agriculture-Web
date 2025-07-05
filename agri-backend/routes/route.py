from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from services.ml_service import MLService
from models.farm import Farm
from services.database import get_db

router = APIRouter()

def get_ml_service():
    return MLService()

@router.post("/predict/yield")
async def predict_yield(farm_id: int, db: Session = Depends(get_db), ml_service: MLService = Depends(get_ml_service)):
    # Get farm data
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    
    # Build features (must match training feature names)
    features = {
        "Soil_Moisture": farm.soil_moisture,
        "Temperature": farm.temperature,
        "Rainfall": farm.rainfall,
        "Crop_Type": farm.crop_type,
        "Market_Price_per_ton": farm.market_price
    }
    
    # Make prediction
    result = ml_service.predict_yield(features)
    return result