import torch
import joblib
import numpy as np
from model_definitions import CropClassifier, YieldPredictor, SustainabilityPredictor
from utils import get_device, to_tensor
from data_preprocessing import preprocess_market_data, preprocess_farmer_data

class AgriPredictorTorch:
    def __init__(self):
        self.device = get_device()
        self.crop_model = self._load_model('crop', CropClassifier)
        self.yield_model = self._load_model('yield', YieldPredictor)
        self.sustainability_model = self._load_model('sustainability', SustainabilityPredictor)
        
    def _load_model(self, model_type, model_class):
        """Load PyTorch model with its dependencies"""
        checkpoint = torch.load(f'models/{model_type}_predictor.pt', map_location=self.device)
        
        if model_type == 'crop':
            model = model_class(
                input_size=17,  # Update based on your combined feature size
                num_classes=len(checkpoint['label_encoder'].classes_)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            return {
                'model': model,
                'label_encoder': checkpoint['label_encoder']
            }
        else:
            model = model_class(input_size=10)  # Update based on farmer feature size
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            return {
                'model': model,
                'y_scaler': checkpoint['y_scaler'],
                'crop_encoder': checkpoint['crop_encoder']
            }
    
    def predict(self, input_data):
        """Make predictions for all models"""
        # Preprocess input data
        market_data = preprocess_market_data(input_data[['Market_Price_per_ton', ...]])  # Add all market columns
        farmer_data, _ = preprocess_farmer_data(input_data, target=None)
        combined_data = np.concatenate([market_data, farmer_data], axis=1)
        
        # Convert to tensor
        combined_tensor = to_tensor(combined_data, self.device)
        farmer_tensor = to_tensor(farmer_data, self.device)
        
        # Crop recommendation
        crop_output = self.crop_model['model'](combined_tensor)
        crop_probs = torch.softmax(crop_output, dim=1)
        crop_pred = torch.argmax(crop_probs, dim=1)
        crop_name = self.crop_model['label_encoder'].inverse_transform([crop_pred.item()])[0]
        
        # Yield prediction
        yield_output = self.yield_model['model'](farmer_tensor)
        yield_scaled = yield_output.cpu().numpy()
        yield_actual = self.yield_model['y_scaler'].inverse_transform(yield_scaled)[0][0]
        
        # Sustainability prediction
        sustain_output = self.sustainability_model['model'](farmer_tensor)
        sustain_scaled = sustain_output.cpu().numpy()
        sustain_actual = self.sustainability_model['y_scaler'].inverse_transform(sustain_scaled)[0][0]
        
        return {
            'recommended_crop': crop_name,
            'crop_confidence': torch.max(crop_probs).item(),
            'predicted_yield': round(yield_actual, 2),
            'sustainability_score': round(sustain_actual, 1),
            'optimal_fertilizer': self._calculate_fertilizer(input_data),
            'water_efficiency': self._calculate_water_efficiency(input_data)
        }
    
    def _calculate_fertilizer(self, input_data):
        """Same as before"""
        # Implementation from previous version
    
    def _calculate_water_efficiency(self, input_data):
        """Same as before"""
        # Implementation from previous version