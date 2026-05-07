import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from modules.waveform_cnn import WaveformCNN
from modules.transformer import TemporalTransformer

class MultimodalGNN(nn.Module):
    def __init__(self, hidden_dim=64, use_cnn=True, use_transformer=True, use_gcn=True, use_dropout=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.use_gcn = use_gcn
        self.use_dropout = use_dropout
        self.hidden_dim = hidden_dim
        
        # Use LazyLinear to automatically adapt to input feature dimension
        self.input_proj = nn.LazyLinear(hidden_dim)
        
        # Short-term pattern extractor
        if self.use_cnn:
            self.cnn_extractor = WaveformCNN(out_channels=hidden_dim)
        else:
            self.raw_proj = nn.LazyLinear(hidden_dim)
            
        # Long-term trend/precursor detector
        if self.use_transformer:
            self.temporal_transformer = TemporalTransformer(input_dim=hidden_dim)
            
        # Spatial relationship detector
        if self.use_gcn:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            
        dropout_p = 0.3 if self.use_dropout else 0.0
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.precursor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch, pos=None):
        # Handle 3D input [batch, seq_len, channels]
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))
        
        # Project to hidden_dim (LazyLinear creates weights on first forward)
        h = self.input_proj(x)
        
        # Skip CNN for now (feature vector is 6000 dims, not raw waveform)
        # If you want to use CNN, you'd need to reshape to [batch, channels, seq_len]
        
        # Process temporal patterns
        if self.use_transformer and hasattr(self, 'temporal_transformer'):
            # Add sequence dimension for transformer
            if h.dim() == 2:
                h = h.unsqueeze(1)
            h = self.temporal_transformer(h)
            if h.dim() == 3:
                h = h.squeeze(1)
        
        # Process spatial graph relationships
        if self.use_gcn:
            h = self.conv1(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)
            
            h = self.conv2(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)
            
            if hasattr(self, 'conv3'):
                h = self.conv3(h, edge_index)
                h = F.relu(h)
        
        # Global pooling to get graph-level representations
        graph_embed = global_mean_pool(h, batch)
        
        if graph_embed.dim() == 1:
            graph_embed = graph_embed.unsqueeze(0)
        
        logits = self.clf_head(graph_embed)
        mag_pred = self.mag_head(graph_embed)
        precursor_prob = torch.sigmoid(self.precursor_head(graph_embed))
        
        return logits, mag_pred, precursor_prob
