import torch
import torch.nn as nn
from utils import get_device

class CropEmbeddingModel(nn.Module):
    """Neural network for crop recommendation with embedding output"""
    def __init__(self, input_size, embedding_size=64):
        super(CropEmbeddingModel, self).__init__()
        self.device = get_device()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return embeddings instead of class probabilities
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

class YieldPredictor(nn.Module):
    """Neural network for yield prediction"""
    def __init__(self, input_size):
        super(YieldPredictor, self).__init__()
        self.device = get_device()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.forward(x)
        return outputs

class SustainabilityPredictor(nn.Module):
    """Neural network for sustainability prediction"""
    def __init__(self, input_size):
        super(SustainabilityPredictor, self).__init__()
        self.device = get_device()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.forward(x)
        return outputs