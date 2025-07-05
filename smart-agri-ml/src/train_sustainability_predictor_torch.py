import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data_preprocessing import preprocess_farmer_data, create_dataloaders
from model_definitions import SustainabilityPredictor
from utils import get_device, EarlyStopping

# Load and preprocess data
df = pd.read_csv('data/raw/farmer_advisor.csv')
X, y, crop_encoder = preprocess_farmer_data(df, target='Sustainability_Score')

# Scale target
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    X, y_scaled, batch_size=16, test_size=0.2
)

# Initialize model
device = get_device()
model = SustainabilityPredictor(input_size=X.shape[1])
model.to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
early_stopping = EarlyStopping(patience=15)

# Training loop
num_epochs = 200
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training phase
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Update scheduler and early stopping
    scheduler.step(val_loss)
    early_stopping(val_loss)
    
    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'y_scaler': y_scaler,
            'crop_encoder': crop_encoder,
        }, 'models/sustainability_predictor.pt')
    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

print("Training complete. Best Validation Loss: {:.4f}".format(best_loss))