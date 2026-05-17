import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """
        Focal Loss for highly imbalanced datasets.
        alpha: Tensor of class weights (e.g., [0.2, 0.8] for 80/20 imbalance)
        gamma: Focusing parameter (typically 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Compute standard cross entropy loss but don't reduce it yet
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)
        # Apply the focal loss formula
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

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
        
        # Classification loss using Focal Loss to fix mode collapse
        # STEAD is approx 81% noise (0), 19% earthquakes (1).
        # We heavily penalize missing the minority class.
        class_weights = torch.tensor([0.2, 0.8], device=logits.device)
        focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        loss_clf = focal_criterion(logits.view(-1, 2), y_true_flat)
        
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
