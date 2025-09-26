import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, d_dense: int, bucket_sizes: list[int], emb_dim: int = 16, mlp: list[int] = [256, 128, 64]):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(b, emb_dim) for b in bucket_sizes])
        fm_in = d_dense + len(bucket_sizes) * emb_dim
        self.linear = nn.Linear(fm_in, 1)
        self.mlp = nn.Sequential(
            nn.Linear(fm_in, mlp[0]),
            nn.ReLU(),
            nn.Linear(mlp[0], mlp[1]),
            nn.ReLU(),
            nn.Linear(mlp[1], mlp[2]),
            nn.ReLU(),
            nn.Linear(mlp[2], 1),
        )

    def forward(self, dense: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        emb = [e(cats[:, i]) for i, e in enumerate(self.embs)]
        x = torch.cat([dense] + emb, dim=1)
        y_linear = self.linear(x)
        y_deep = self.mlp(x)
        y = torch.sigmoid(y_linear + y_deep)
        return y

