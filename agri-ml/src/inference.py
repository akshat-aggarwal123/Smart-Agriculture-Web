import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

class YieldPredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model("models/yield_prediction/saved_model/")
        self.preprocessor = joblib.load("models/yield_prediction/preprocessor.joblib")
    
    def predict(self, input_data):
        """
        input_data: Dictionary with features matching the original dataset
        Example:
        {
            "Soil_Moisture": 45.2,
            "Temperature": 28.5,
            "Crop_Type": "Wheat",
            "Market_Price_per_ton": 250
        }
        """
        df = pd.DataFrame([input_data])
        processed = self.preprocessor.transform(df)
        prediction = self.model.predict(processed, verbose=0)
        return float(prediction[0][0])

# Example usage
if __name__ == "__main__":
    predictor = YieldPredictor()
    
    sample_input = {
        "Soil_Moisture": 45.2,
        "Temperature": 28.5,
        "Crop_Type": "Wheat",
        "Market_Price_per_ton": 250
    }
    
    yield_prediction = predictor.predict(sample_input)
    print(f"Predicted Yield: {yield_prediction:.2f} tons")
