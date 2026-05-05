import torch
import numpy as np
from torch_geometric.loader import DataLoader
from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error

def evaluate_model(batch_size=4096, model_path='earthquake_gnn.pth', use_cnn=True, use_transformer=True, use_gcn=True, use_dropout=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        dataset = STEADGraphDataset()
    except Exception as e:
        print(f"Dataset loading error: {e}")
        return

    num_workers = 16 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'
    
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = MultimodalGNN(hidden_dim=64, use_cnn=use_cnn, use_transformer=use_transformer, use_gcn=use_gcn, use_dropout=use_dropout).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Could not load model weights from {model_path}.")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    all_mag_preds = []
    all_mag_true = []
    all_pre_preds = []
    all_pre_true = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits, mag_pred, pre_prob = model(data.x, data.edge_index, data.batch, data.pos)
            
            # Event Detection
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
            # Precursor Detection (Threshold at 0.5)
            pre_preds = (pre_prob > 0.5).float()
            all_pre_preds.extend(pre_preds.cpu().numpy())
            all_pre_true.extend(data.precursor.cpu().numpy())
            
            # Magnitude Estimation
            mask = data.y.view(-1) == 1
            if mask.sum() > 0:
                all_mag_preds.extend(mag_pred[mask].squeeze(-1).cpu().numpy())
                all_mag_true.extend(data.mag[mask].squeeze(-1).cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    pre_acc = accuracy_score(all_pre_true, all_pre_preds)
    
    print("\n--- Evaluation Results ---")
    print(f"Event Detection Accuracy: {acc*100:.2f}%")
    print(f"Event Detection F1-Score: {f1:.4f}")
    print(f"Precursor Prediction Accuracy: {pre_acc*100:.2f}%")
    
    if len(all_mag_true) > 0:
        mse = mean_squared_error(all_mag_true, all_mag_preds)
        print(f"Magnitude MSE: {mse:.4f}")
    print("--------------------------")
