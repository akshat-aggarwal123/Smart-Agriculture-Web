import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import get_device

def preprocess_market_data(df):
    """Preprocess market researcher dataset for PyTorch"""
    # Encode seasonal factor
    seasonal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Seasonal_Factor'] = df['Seasonal_Factor'].map(seasonal_map)
    
    # Select features and target
    features = ['Demand_index', 'Supply_index', 'Competitor_Price_per_ton', 
                'Economic_Indicator', 'Weather_Impact_Score', 'Seasonal_Factor',
                'Consumer_Trend_index']
    X = df[features].values
    return X

def preprocess_farmer_data(df, target=None):
    """Preprocess farmer advisor dataset for PyTorch"""
    # One-hot encode crop type
    crop_encoder = OneHotEncoder(sparse_output=False)
    crop_encoded = crop_encoder.fit_transform(df[['Crop_Type']])
    crop_cols = [f'Crop_{c}' for c in crop_encoder.categories_[0]]
    crop_df = pd.DataFrame(crop_encoded, columns=crop_cols)
    
    # Feature selection
    features = ['Soil_ph', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
                'Fertilizer_Usage_kg', 'Pesticide_Usage_kg']
    
    # Combine features
    X = pd.concat([df[features], crop_df], axis=1).values
    
    if target:
        y = df[target].values
        return X, y, crop_encoder
    return X, crop_encoder

def create_dataloaders(X, y, batch_size=32, test_size=0.2, shuffle=True):
    """Create PyTorch DataLoaders for training"""
    from torch.utils.data import TensorDataset, DataLoader
    from utils import to_tensor
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    
    # Convert to tensors
    X_train_tensor = to_tensor(X_train)
    y_train_tensor = to_tensor(y_train)
    X_test_tensor = to_tensor(X_test)
    y_test_tensor = to_tensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader