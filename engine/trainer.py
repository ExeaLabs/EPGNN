import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN
from losses.custom_losses import EPGNNLoss
from tqdm import tqdm

import torch.backends.cudnn as cudnn

def train_model(epochs=5, batch_size=4096, use_cnn=True, use_transformer=True, use_gcn=True, use_dropout=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        cudnn.benchmark = True
        print("Enabled cuDNN autotuning for high performance.")
        
    try:
        dataset = STEADGraphDataset()
    except Exception as e:
        print(f"Dataset loading error: {e}")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    num_workers = 16 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = MultimodalGNN(hidden_dim=64, use_cnn=use_cnn, use_transformer=use_transformer, use_gcn=use_gcn, use_dropout=use_dropout).to(device)

    criterion = EPGNNLoss(mag_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, mag_pred, pre_prob = model(data.x, data.edge_index, data.batch, data.pos)
                    loss = criterion(logits, mag_pred, pre_prob, data.y, data.mag, data.precursor)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, mag_pred, pre_prob = model(data.x, data.edge_index, data.batch, data.pos)
                loss = criterion(logits, mag_pred, pre_prob, data.y, data.mag, data.precursor)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")
            
    torch.save(model.state_dict(), 'earthquake_gnn.pth')
    print("Model saved to earthquake_gnn.pth")
