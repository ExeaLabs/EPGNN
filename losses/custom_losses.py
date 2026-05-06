import torch
import torch.nn as nn
import torch.nn.functional as F

class EPGNNLoss(nn.Module):
    def __init__(self, mag_weight=0.1, pre_weight=1.0):
        super().__init__()
        self.mag_weight = mag_weight
        self.pre_weight = pre_weight
        
    def forward(self, logits, mag_pred, pre_prob, y_true, mag_true, pre_true):
        # Flatten all target tensors to 1D
        y_true_flat = y_true.view(-1).long()
        pre_true_flat = pre_true.view(-1).float()
        mag_true_flat = mag_true.view(-1).float()
        
        # Flatten predictions
        mag_pred_flat = mag_pred.view(-1)
        pre_prob_flat = pre_prob.view(-1)
        
        # Classification loss (with class weights for imbalance)
        class_weights = torch.tensor([1.0, 3.0], device=logits.device)  # Adjust based on your data
        loss_clf = F.cross_entropy(logits.view(-1, 2), y_true_flat, weight=class_weights)
        
        # Precursor loss (binary classification)
        loss_pre = F.binary_cross_entropy_with_logits(pre_prob_flat, pre_true_flat)
        
        # Base loss
        loss = loss_clf + self.pre_weight * loss_pre
        
        # Magnitude loss (only for earthquake events)
        earthquake_mask = (y_true_flat == 1)
        if earthquake_mask.any():
            loss_mag = F.mse_loss(
                mag_pred_flat[earthquake_mask], 
                mag_true_flat[earthquake_mask]
            )
            loss = loss + self.mag_weight * loss_mag
        
        return loss
