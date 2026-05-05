"""
EPGNN Exhaustive Ablation Suite
================================
Runs all combinations of:
  - CNN (on/off)
  - Transformer (on/off)
  - GCN (on/off)
  - Dropout (on/off)

Skips degenerate models where all core compute modules are disabled.

Outputs:
  - ablation_results.json (paper-ready structured results)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import json
import time
from itertools import product
from dataclasses import dataclass, asdict
from typing import Optional, List

from data.dataset import STEADGraphDataset
from modules.waveform_cnn import WaveformCNN
from modules.transformer import TemporalTransformer


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class AblatedGNN(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        use_cnn=True,
        use_transformer=True,
        use_gcn=True,
        use_dropout=True,
    ):
        super().__init__()

        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.use_gcn = use_gcn

        if use_cnn:
            self.cnn = WaveformCNN(out_channels=hidden_dim)
        else:
            self.raw_proj = nn.LazyLinear(hidden_dim)

        if use_transformer:
            self.tx = TemporalTransformer(input_dim=hidden_dim)

        if use_gcn:
            self.gcn1 = GCNConv(hidden_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        drop_p = 0.3 if use_dropout else 0.0

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, batch, pos=None):
        if self.use_cnn:
            h = self.cnn(x)
        else:
            h = self.raw_proj(x)

        if self.use_transformer:
            h = self.tx(h.unsqueeze(1))

        if self.use_gcn:
            h = F.relu(self.gcn1(h, edge_index))
            h = F.relu(self.gcn2(h, edge_index))

        g = global_mean_pool(h, batch)
        return self.head(g)


# ─────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────

@dataclass
class Result:
    name: str
    use_cnn: bool
    use_transformer: bool
    use_gcn: bool
    use_dropout: bool

    accuracy: float
    params: int
    collapsed: bool
    runtime_sec: float


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    preds_all = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            preds = logits.argmax(dim=1)

            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

            preds_all.extend(preds.cpu().numpy())

    acc = correct / total
    collapsed = len(set(preds_all)) == 1
    return acc, collapsed


# ─────────────────────────────────────────────
# Exhaustive ablations
# ─────────────────────────────────────────────

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = STEADGraphDataset(
        metadata_path="metadata_clean.csv",
        hdf5_path="mock_waveforms.hdf5",
    )

    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    MODEL_WEIGHTS = "earthquake_gnn.pth"

    combos = list(product([False, True], repeat=4))

    results: List[Result] = []

    print(f"Running {len(combos)} ablations...\n")

    for i, (cnn, tx, gcn, drop) in enumerate(combos):

        # skip degenerate model
        if not (cnn or tx or gcn):
            continue

        name = f"CNN={cnn}_TX={tx}_GCN={gcn}_DO={drop}"
        print(f"[{i+1}/{len(combos)}] {name}")

        model = AblatedGNN(
            hidden_dim=64,
            use_cnn=cnn,
            use_transformer=tx,
            use_gcn=gcn,
            use_dropout=drop,
        ).to(device)

        # load partial pretrained weights
        state = torch.load(MODEL_WEIGHTS, map_location=device)
        model.load_state_dict(state, strict=False)

        t0 = time.time()
        acc, collapsed = evaluate(model, loader, device)
        dt = time.time() - t0

        results.append(Result(
            name=name,
            use_cnn=cnn,
            use_transformer=tx,
            use_gcn=gcn,
            use_dropout=drop,
            accuracy=acc,
            params=count_params(model),
            collapsed=collapsed,
            runtime_sec=dt,
        ))

        print(f"    acc={acc:.4f}, time={dt:.1f}s")

    # ─────────────────────────────────────────────
    # Save JSON (paper-ready)
    # ─────────────────────────────────────────────

    output = {
        "dataset": "STEAD",
        "model": "EPGNN Ablation Suite",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_runs": len(results),
        "results": [asdict(r) for r in results],
    }

    with open("ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved → ablation_results.json")


if __name__ == "__main__":
    run_all()
