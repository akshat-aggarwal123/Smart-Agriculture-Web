import joblib
import tensorflow as tf
import numpy as np
from app.config import settings

class MLService:
    def __init__(self):
        self.models = {
            "yield": joblib.load(f"{settings.ML_MODEL_PATH}/yield_model.pkl"),
            "irrigation": tf.keras.models.load_model(f"{settings.ML_MODEL_PATH}/irrigation_model.h5")
        }
    
    def predict_yield(self, features: dict) -> dict:
        features_array = np.array([list(features.values())])
        prediction = self.models["yield"].predict(features_array)
        
        return {
            "predicted_yield": float(prediction[0]),
            "confidence": 0.92
        }