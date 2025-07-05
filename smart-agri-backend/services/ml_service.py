import torch
import joblib
import numpy as np
from config import settings
from sklearn.compose import ColumnTransformer

class MLService:
    def __init__(self):
        # Load PyTorch model
        try:
            self.model = torch.jit.load(f"{settings.ML_MODEL_PATH}/final_model.pth")
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")
        
        # Load preprocessor
        try:
            self.preprocessor = joblib.load(f"{settings.ML_MODEL_PATH}/preprocessor.joblib")
        except Exception as e:
            raise RuntimeError(f"Failed to load preprocessor: {e}")
    
    def predict_yield(self, features: dict) -> dict:
        """
        Predict crop yield using trained PyTorch model
        Features must match training features:
        - Soil_Moisture
        - Temperature
        - Rainfall
        - Crop_Type
        - Market_Price_per_ton
        """
        try:
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame([features])
            
            # Apply preprocessing
            processed = self.preprocessor.transform(df)
            
            # Convert to tensor
            input_tensor = torch.tensor(processed.toarray(), dtype=torch.float32)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(input_tensor).item()
            
            return {
                "predicted_yield_tons": prediction,
                "confidence": 0.92  # Replace with real confidence if available
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")