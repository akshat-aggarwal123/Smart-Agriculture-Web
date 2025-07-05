import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_farmer_data, create_dataloaders
from model_definitions import SustainabilityPredictor
from utils import get_device
from triplet_loss import TripletLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('models', exist_ok=True)

# Load and preprocess data
df = pd.read_csv('data/raw/farmer_advisor.csv')
X, y, crop_encoder = preprocess_farmer_data(df, target='Sustainability_Score')

# Scale target
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

# Create dataloader with all data
train_loader, _ = create_dataloaders(
    X, y_scaled, 
    batch_size=len(X),  # Single batch with all data
    test_size=0.0,  # Use all data for training
    is_classification=False
)

# Initialize model
device = get_device()
model = SustainabilityPredictor(input_size=X.shape[1])
model.to(device)

# Triplet loss
criterion = TripletLoss(margin=1.0)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Get all data in one batch
all_data, all_labels = next(iter(train_loader))
all_data = all_data.to(device)
all_labels = all_labels.to(device)

# Flatten labels to 1D tensor
all_labels = all_labels.view(-1)

# Training loop
num_epochs = 1000
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    
    # Generate triplets
    anchors, positives, negatives = [], [], []
    
    # For each sample as anchor
    for i in range(len(all_data)):
        anchor = all_data[i]
        label = all_labels[i]
        
        # Find positive samples (similar sustainability)
        # Use a more permissive threshold: within 1 standard deviation
        similar_mask = torch.abs(all_labels - label) < 1.0
        similar_mask[i] = False  # Exclude self
        similar_indices = torch.where(similar_mask)[0]
        
        # Find negative samples (different sustainability)
        # Use a more permissive threshold: beyond 0.5 standard deviations
        different_mask = torch.abs(all_labels - label) >= 0.5
        different_indices = torch.where(different_mask)[0]
        
        # Only proceed if we have both positives and negatives
        if len(similar_indices) > 0 and len(different_indices) > 0:
            # Randomly select one positive and one negative
            positive_idx = random.choice(similar_indices.cpu().numpy())
            negative_idx = random.choice(different_indices.cpu().numpy())
            
            anchors.append(anchor)
            positives.append(all_data[positive_idx])
            negatives.append(all_data[negative_idx])
    
    # Skip if no triplets generated
    if len(anchors) == 0:
        print(f"Epoch {epoch+1}: No triplets generated - trying next epoch")
        continue
    
    # Convert to tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    
    # Forward pass
    anchor_emb = model(anchors)
    positive_emb = model(positives)
    negative_emb = model(negatives)
    
    # Calculate loss
    loss = criterion(anchor_emb, positive_emb, negative_emb)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Calculate accuracy
    with torch.no_grad():
        # Compute pairwise distances
        dist_pos = torch.norm(anchor_emb - positive_emb, dim=1)
        dist_neg = torch.norm(anchor_emb - negative_emb, dim=1)
        
        # Count correct triplets
        correct = torch.sum(dist_pos < dist_neg).item()
        accuracy = correct / len(anchors)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {loss.item():.4f} | "
              f"Triplet Acc: {accuracy:.4f} | "
              f"Triplets: {len(anchors)} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'y_scaler': y_scaler,
            'crop_encoder': crop_encoder,
        }, 'models/sustainability_predictor.pt')

print("Training complete. Best Loss: {:.4f}".format(best_loss))