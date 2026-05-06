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
    
    # Load dataset
    dataset = STEADGraphDataset()
    
    # Ensure data types are correct in the dataset
    for i in range(len(dataset)):
        data = dataset[i]
        data.x = data.x.float()
        data.y = data.y.long()  # Classification labels should be long
        data.mag = data.mag.float().view(-1, 1)
        data.precursor = data.precursor.float().view(-1, 1)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = MultimodalGNN(
        hidden_dim=64, 
        use_cnn=use_cnn, 
        use_transformer=use_transformer, 
        use_gcn=use_gcn, 
        use_dropout=use_dropout
    ).to(device)
    
    criterion = EPGNNLoss(mag_weight=0.1, precursor_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Increased learning rate from 1e-5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            
            # Forward pass
            logits, mag_pred, precursor_prob = model(data.x, data.edge_index, data.batch, data.pos)
            
            # Compute loss
            loss = criterion(logits, mag_pred, precursor_prob, data.y, data.mag, data.precursor)
            
            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print batch loss occasionally
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
        
        # Validation (optional)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                logits, mag_pred, precursor_prob = model(data.x, data.edge_index, data.batch, data.pos)
                loss = criterion(logits, mag_pred, precursor_prob, data.y, data.mag, data.precursor)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
    torch.save(model.state_dict(), 'earthquake_gnn.pth')
    print("Model saved to earthquake_gnn.pth")
