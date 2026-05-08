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

        # ✅ Dynamic input projection – works with any feature size (e.g., 6000)
        self.input_proj = nn.LazyLinear(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Optional CNN (only for raw waveforms, will be bypassed for 6000‑dim features)
        if self.use_cnn:
            self.cnn_extractor = WaveformCNN(out_channels=hidden_dim)
        else:
            self.raw_proj = nn.LazyLinear(hidden_dim)

        # Transformer for temporal patterns
        if self.use_transformer:
            self.temporal_transformer = TemporalTransformer(input_dim=hidden_dim)

        # GCN layers
        if self.use_gcn:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)   # optional

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
        # x shape: [num_nodes, input_features] – works for any input_dim
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))

        # Guard against non-finite values and extreme feature scales.
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - x_mean) / x_std

        # Project to hidden_dim (LazyLinear creates weights on first forward)
        h = self.input_proj(x)
        h = self.input_norm(h)

        # CNN branch is skipped because input features are not raw 3‑channel waveforms
        # (you could add a reshape here if needed, but for 6000‑dim it's fine to skip)

        # Transformer (adds dummy sequence dimension)
        if self.use_transformer and hasattr(self, 'temporal_transformer'):
            h = h.unsqueeze(1)          # [num_nodes, 1, hidden_dim]
            h = self.temporal_transformer(h)
            h = h.squeeze(1)

        # GCN layers
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

        # Global pooling to graph level
        graph_embed = global_mean_pool(h, batch)
        if graph_embed.dim() == 1:
            graph_embed = graph_embed.unsqueeze(0)

        logits = self.clf_head(graph_embed)
        mag_pred = self.mag_head(graph_embed)
        precursor_logits = self.precursor_head(graph_embed)

        return logits, mag_pred, precursor_logits
