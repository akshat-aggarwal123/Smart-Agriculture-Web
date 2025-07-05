import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import preprocess_market_data, preprocess_farmer_data, create_dataloaders
from model_definitions import CropClassifier
from utils import get_device, EarlyStopping

# Load and preprocess data
market_df = pd.read_csv('data/raw/market_researcher.csv')
farmer_df = pd.read_csv('data/raw/farmer_advisor.csv')

# Preprocess data
market_features = preprocess_market_data(market_df)
farmer_features, crop_encoder = preprocess_farmer_data(farmer_df)

# Combine datasets
combined = np.concatenate([market_features, farmer_features], axis=1)
y = market_df['Product'].values

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    combined, y_encoded, batch_size=32, test_size=0.2
)

# Initialize model
device = get_device()
model = CropClassifier(input_size=combined.shape[1], num_classes=len(label_encoder.classes_))
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
early_stopping = EarlyStopping(patience=10)

# Training loop
num_epochs = 100
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training phase
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(test_loader)
    val_acc = correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # Update scheduler and early stopping
    scheduler.step(val_loss)
    early_stopping(val_loss)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'acc': val_acc,
            'label_encoder': label_encoder,
        }, 'models/crop_recommender.pt')
    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

print("Training complete. Best Validation Accuracy: {:.4f}".format(best_acc))