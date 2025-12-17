"""CNN Encoder for 8×16 character images."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input: 8×16 (W×H), 1:2 aspect ratio like terminal chars
CELL_W, CELL_H = 8, 16


class CharEncoder(nn.Module):
    """CNN: (B, 1, 16, 8) -> (B, embedding_dim) normalized embeddings."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        # 8×16 -> 4×8 -> 2×4 -> 1×2
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        # After pooling: 1×2 × 128 = 256
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (CELL_H // 8) * (CELL_W // 8), 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.fc(self.conv(x)), p=2, dim=1)


def contrastive_loss(anchor, positive, temperature=0.1):
    """NT-Xent contrastive loss."""
    B = anchor.size(0)
    emb = torch.cat([anchor, positive], dim=0)
    sim = emb @ emb.t() / temperature
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(anchor.device)
    mask = torch.eye(2 * B, dtype=torch.bool, device=anchor.device)
    return F.cross_entropy(sim.masked_fill(mask, float('-inf')), labels)


if __name__ == '__main__':
    m = CharEncoder(64)
    x = torch.randn(4, 1, CELL_H, CELL_W)
    print(f"Input: {x.shape}, Output: {m(x).shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
