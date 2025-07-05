import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(data, device=None, dtype=torch.float32):
    if not device:
        device = get_device()
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype).to(device)
    elif isinstance(data, torch.Tensor):
        tensor = data.to(dtype).to(device)
    else:
        raise ValueError("Unsupported data type")
    return tensor

def inverse_scale(scaler, tensor):
    return torch.from_numpy(scaler.inverse_transform(tensor.cpu().numpy())).float()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0