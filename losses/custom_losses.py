import torch
import torch.nn as nn

class EPGNNLoss(nn.Module):
    def __init__(self, mag_weight=0.1, pre_weight=1.0):
        super().__init__()
        self.clf_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        self.pre_criterion = nn.BCEWithLogitsLoss()
        self.mag_weight = mag_weight
        self.pre_weight = pre_weight

    def forward(self, logits, mag_pred, pre_prob, y_true, mag_true, pre_true):
        loss_clf = self.clf_criterion(logits, y_true.view(-1))
        loss_pre = self.pre_criterion(pre_prob.view(-1), pre_true.view(-1))
        
        mask = y_true.view(-1) == 1
        loss = loss_clf + self.pre_weight * loss_pre
        
        if mask.sum() > 0:
            loss_mag = self.reg_criterion(mag_pred[mask].squeeze(-1), mag_true[mask].squeeze(-1))
            loss += self.mag_weight * loss_mag
            
        return loss
