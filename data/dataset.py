import torch
import numpy as np
import seisbench.data as sbd
from torch_geometric.data import Dataset, Data

class STEADGraphDataset(Dataset):
    def __init__(self, root='.', transform=None, pre_transform=None, **kwargs):
        super().__init__(root, transform, pre_transform)
        # Initialize SeisBench STEAD dataset (downloads automatically if missing)
        self.stead = sbd.STEAD(download=True)
        
        # Spatial graph: 3 components (E, N, Z) treated as 3 fully connected nodes
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1]
        ], dtype=torch.long)

    def len(self):
        return len(self.stead)

    def get(self, idx):
        # Fetch waveform and metadata from SeisBench
        waveform, metadata = self.stead.get_sample(idx)
        
        # waveform shape is usually (3, samples). PyTorch Geometric treats this as 3 nodes with `samples` features.
        x = torch.tensor(waveform, dtype=torch.float)
        
        # Determine label (1 for earthquake, 0 for noise)
        trace_category = metadata.get("trace_category", "noise")
        label = 1 if "earthquake" in trace_category else 0
        y = torch.tensor([label], dtype=torch.long)
        
        # Source magnitude (fallback to 0.0 if NaN/None)
        mag_val = metadata.get("source_magnitude")
        if mag_val is None or np.isnan(mag_val):
            mag_val = 0.0
        mag = torch.tensor([mag_val], dtype=torch.float)
        
        # Receiver coordinates (fallback to 0.0 if missing)
        lat = metadata.get("receiver_latitude", 0.0)
        lon = metadata.get("receiver_longitude", 0.0)
        pos = torch.tensor([[lat, lon]], dtype=torch.float)
        
        # Precursor flag (mocked for now, as STEAD natively doesn't label precursors)
        precursor = torch.tensor([0.0], dtype=torch.float)
        
        # Pack into PyTorch Geometric Data object
        data = Data(x=x, edge_index=self.edge_index, y=y, pos=pos, mag=mag, precursor=precursor)
        
        return data

if __name__ == '__main__':
    ds = STEADGraphDataset()
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    print(f"Sample x shape: {sample.x.shape}")

