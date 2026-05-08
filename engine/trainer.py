import torch
import torch.optim as optim
from pathlib import Path
from torch_geometric.loader import DataLoader
from data.dataset import STEADGraphDataset
from models.gnn import MultimodalGNN
from losses.custom_losses import EPGNNLoss
from tqdm import tqdm

import torch.backends.cudnn as cudnn

def train_model(
    epochs=5,
    batch_size=4096,
    use_cnn=True,
    use_transformer=True,
    use_gcn=True,
    use_dropout=True,
    split_cache_path="cache/train_val_split.pt",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = STEADGraphDataset()

    # Cache train/val split indices so repeated runs skip split rebuild and stay reproducible.
    split_cache = Path(split_cache_path)
    split_cache.parent.mkdir(parents=True, exist_ok=True)
    expected_len = len(dataset)

    train_indices = None
    val_indices = None
    if split_cache.exists():
        try:
            cached = torch.load(split_cache)
            if cached.get("dataset_len") == expected_len:
                train_indices = cached["train_indices"]
                val_indices = cached["val_indices"]
                print(f"Loaded cached split from {split_cache}")
        except Exception:
            # If cache is corrupted or incompatible, rebuild below.
            train_indices = None
            val_indices = None

    if train_indices is None or val_indices is None:
        train_size = int(0.8 * expected_len)
        perm = torch.randperm(expected_len)
        train_indices = perm[:train_size].tolist()
        val_indices = perm[train_size:].tolist()
        torch.save(
            {
                "dataset_len": expected_len,
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
            split_cache,
        )
        print(f"Saved cached split to {split_cache}")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # One-time batch diagnostics to verify tensor ranges and shapes.
    for data in train_loader:
        print(f"Data.x shape: {data.x.shape}")
        print(f"Data.x dtype: {data.x.dtype}")
        print(f"Data.x min/max: {data.x.min()}/{data.x.max()}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Batch shape: {data.batch.shape}")
        break
    
    model = MultimodalGNN(
        hidden_dim=64, 
        use_cnn=use_cnn, 
        use_transformer=use_transformer, 
        use_gcn=use_gcn, 
        use_dropout=use_dropout
    ).to(device)
    
    criterion = EPGNNLoss(mag_weight=0.1, pre_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        valid_train_batches = 0
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            data.x = torch.nan_to_num(data.x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            data.mag = torch.nan_to_num(data.mag.float(), nan=0.0, posinf=10.0, neginf=-10.0).view(-1, 1)
            data.precursor = torch.nan_to_num(data.precursor.float(), nan=0.0, posinf=1.0, neginf=0.0).view(-1, 1)
            
            # Forward pass
            logits, mag_pred, precursor_logits = model(data.x, data.edge_index, data.batch, data.pos)

            if not (torch.isfinite(logits).all() and torch.isfinite(mag_pred).all() and torch.isfinite(precursor_logits).all()):
                print(f"Warning: Non-finite model output at batch {batch_idx}, skipping...")
                continue
            
            # Compute loss
            loss = criterion(logits, mag_pred, precursor_logits, data.y, data.mag, data.precursor)
            
            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                print(f"Warning: Non-finite gradient norm at batch {batch_idx}, skipping optimizer step...")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            
            train_loss += loss.item()
            valid_train_batches += 1
            
            # Print batch loss occasionally
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        avg_train_loss = train_loss / max(valid_train_batches, 1)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
        
        # Validation (optional)
        model.eval()
        val_loss = 0.0
        valid_val_batches = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = torch.nan_to_num(data.x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
                data.mag = torch.nan_to_num(data.mag.float(), nan=0.0, posinf=10.0, neginf=-10.0).view(-1, 1)
                data.precursor = torch.nan_to_num(data.precursor.float(), nan=0.0, posinf=1.0, neginf=0.0).view(-1, 1)

                logits, mag_pred, precursor_logits = model(data.x, data.edge_index, data.batch, data.pos)
                if not (torch.isfinite(logits).all() and torch.isfinite(mag_pred).all() and torch.isfinite(precursor_logits).all()):
                    continue

                loss = criterion(logits, mag_pred, precursor_logits, data.y, data.mag, data.precursor)
                if not torch.isfinite(loss):
                    continue
                val_loss += loss.item()
                valid_val_batches += 1
        
        avg_val_loss = val_loss / max(valid_val_batches, 1)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
    torch.save(model.state_dict(), 'earthquake_gnn.pth')
    print("Model saved to earthquake_gnn.pth")
