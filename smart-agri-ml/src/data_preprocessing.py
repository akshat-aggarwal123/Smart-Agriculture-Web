import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from utils import get_device
import torch  # ADD THIS IMPORT

def preprocess_market_data(df):
    """Preprocess market researcher dataset"""
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Encode seasonal factor
    seasonal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Seasonal_Factor'] = df['Seasonal_Factor'].map(seasonal_map)
    
    # Select features and target
    features = [
        'Demand_Index', 
        'Supply_Index', 
        'Competitor_Price_per_ton',
        'Economic_Indicator', 
        'Weather_Impact_Score', 
        'Seasonal_Factor',
        'Consumer_Trend_Index'
    ]
    
    X = df[features].values
    return X

def preprocess_farmer_data(df, target=None):
    """Preprocess farmer advisor dataset"""
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # One-hot encode crop type
    crop_encoder = OneHotEncoder(sparse_output=False)
    crop_encoded = crop_encoder.fit_transform(df[['Crop_Type']])
    crop_cols = [f'Crop_{c}' for c in crop_encoder.categories_[0]]
    crop_df = pd.DataFrame(crop_encoded, columns=crop_cols)
    
    # Feature selection
    features = [
        'Soil_pH', 
        'Soil_Moisture', 
        'Temperature_C', 
        'Rainfall_mm',
        'Fertilizer_Usage_kg', 
        'Pesticide_Usage_kg'
    ]
    
    # Combine features
    X = pd.concat([df[features], crop_df], axis=1).values
    
    if target:
        y = df[target].values
        return X, y, crop_encoder
    return X, crop_encoder

def create_dataloaders(X, y, batch_size=32, test_size=0.2, shuffle=True, is_classification=False):
    """Create PyTorch DataLoaders for training"""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from utils import to_tensor
    
    # Handle case when using all data for training
    if test_size == 0.0:
        # Convert to tensors
        X_tensor = to_tensor(X)
        y_dtype = torch.long if is_classification else torch.float32
        y_tensor = to_tensor(y, dtype=y_dtype)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, None
    
    # Split data normally
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    
    # Convert to tensors
    X_train_tensor = to_tensor(X_train)
    y_dtype = torch.long if is_classification else torch.float32
    y_train_tensor = to_tensor(y_train, dtype=y_dtype)
    X_test_tensor = to_tensor(X_test)
    y_test_tensor = to_tensor(y_test, dtype=y_dtype)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader