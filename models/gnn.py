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
        
        # Input projection - CRITICAL: Need to handle input dimension properly
        self.input_proj = nn.Linear(3, hidden_dim)  # Assuming 3 channels (Z,N,E)
        
        # Short-term pattern extractor
        if self.use_cnn:
            self.cnn_extractor = WaveformCNN(out_channels=hidden_dim)
        else:
            self.raw_proj = nn.Linear(3, hidden_dim)
            
        # Long-term trend/precursor detector
        if self.use_transformer:
            self.temporal_transformer = TemporalTransformer(input_dim=hidden_dim)
            
        # Spatial relationship detector
        if self.use_gcn:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Added deeper GCN
            
        dropout_p = 0.3 if self.use_dropout else 0.0
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 2)  # Detection (earthquake or not)
        )
        
        self.precursor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Probability of event in next hour
        )
        
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Magnitude estimation
        )
        
        # Initialize weights for better training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to avoid vanishing gradients"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, GCNConv):
                nn.init.xavier_uniform_(module.lin.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, edge_index, batch, pos=None):
        """
        Args:
            x: Node features [num_nodes, 3] - waveform data (Z,N,E channels)
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            pos: Optional positional encoding (not used in this version)
        """
        # Ensure input has correct shape
        if x.dim() == 3:
            # If x is [batch, seq_len, channels], flatten the sequence dimension
            x = x.view(-1, x.size(-1))
        
        # Project input to hidden dimension if needed
        if x.size(-1) != 64:  # If not already at hidden_dim
            if hasattr(self, 'input_proj'):
                x = self.input_proj(x)
        
        # 1. Extract features from waveforms
        if self.use_cnn:
            # CNN expects [batch, channels, sequence_length]
            # Reshape if necessary
            if x.dim() == 2:
                # Assume [num_nodes, features] - need to reshape to [batch, channels, seq_len]
                # For now, just pass through raw_proj
                h = self.raw_proj(x) if hasattr(self, 'raw_proj') else x
            else:
                h = self.cnn_extractor(x)
        else:
            h = self.raw_proj(x) if hasattr(self, 'raw_proj') else x
            
        # Ensure h is at hidden_dim
        if h.size(-1) != 64:
            h = self.raw_proj(x)
            
        # 2. Process temporal patterns
        if self.use_transformer and hasattr(self, 'temporal_transformer'):
            # Add sequence dimension for transformer [batch, seq_len, features]
            if h.dim() == 2:
                h = h.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            h = self.temporal_transformer(h)
            # Remove sequence dimension if added
            if h.dim() == 3:
                h = h.squeeze(1)
            
        # 3. Process spatial graph relationships
        if self.use_gcn:
            # First GCN layer
            h = self.conv1(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)
            
            # Second GCN layer
            h = self.conv2(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)
            
            # Third GCN layer
            if hasattr(self, 'conv3'):
                h = self.conv3(h, edge_index)
                h = F.relu(h)
            
        # Global pooling to get graph-level representations
        graph_embed = global_mean_pool(h, batch)
        
        # Ensure graph_embed has correct shape
        if graph_embed.dim() == 1:
            graph_embed = graph_embed.unsqueeze(0)
            
        # Classification head (earthquake detection)
        logits = self.clf_head(graph_embed)
        
        # Magnitude prediction head (only for earthquake samples)
        mag_pred = self.mag_head(graph_embed)
        
        # Precursor probability (earthquake in next hour)
        precursor_prob = torch.sigmoid(self.precursor_head(graph_embed))
        
        return logits, mag_pred, precursor_prob
