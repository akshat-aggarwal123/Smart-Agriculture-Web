import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_distance = (anchor - positive).pow(2).sum(1)  # Euclidean distance squared
        neg_distance = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()