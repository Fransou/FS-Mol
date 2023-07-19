import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class SimpleShotConfig:
    lmbd_entropy: float = 0.3
    temperature: float = 0.8
    learning_rate: float = 1e-3
    epochs: int = 10
    clip_grad_norm: float = 1.0
    center_data: bool = True
    normalize_norm: bool = True
    distance: str = "cosine"
    bias: bool = True
    out_shape: int = 64


class SimpleshotNet(nn.Module):
    def __init__(self, X_support_pos, X_support_neg, config: SimpleShotConfig):
        super().__init__()
        self.config = config
        inp_shape = X_support_pos.shape[1]
        self.inp_layer = nn.Linear(
            inp_shape, self.config.out_shape, bias=self.config.bias
        )
        self.lmbd_entropy = config.lmbd_entropy
        self.temperature = config.temperature
        self.bce = nn.BCELoss()
        self.entropy = ShannonEntroy()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.X_support_pos = torch.tensor(X_support_pos, dtype=torch.float32)
        self.X_support_neg = torch.tensor(X_support_neg, dtype=torch.float32)

    def forward(self, x):
        x_support_pos = self.inp_layer(self.X_support_pos)
        x_support_neg = self.inp_layer(self.X_support_neg)
        x_query = self.inp_layer(x)

        prot_pos = x_support_pos.mean(axis=0)
        prot_neg = x_support_neg.mean(axis=0)

        if self.config.distance == "cosine":
            x_pos = self.temperature * (self.cos(x_query, prot_pos))
            x_neg = self.temperature * (self.cos(x_query, prot_neg))
            x = torch.sigmoid(x_pos) / (torch.sigmoid(x_pos) + torch.sigmoid(x_neg))
        elif self.config.distance == "euclidean":
            x_pos = self.temperature * (
                torch.norm(x_query- prot_pos, dim=1)
            )
            x_neg = self.temperature * (
                torch.norm(x_query- prot_neg, dim=1)
            )
            x = torch.exp(-x_pos) / (torch.exp(-x_pos) + torch.exp(-x_neg))

        return x

    def predict(self, x):
        return self.forward(x)

    def get_loss(self, x_support, y_support, x_query):
        y_support_pred = self.forward(x_support)
        cross_entopy_support = self.bce(y_support_pred, y_support)
        y_query_pred = self.forward(x_query)
        entropy_query = self.entropy(y_query_pred)

        return (cross_entopy_support + self.lmbd_entropy * entropy_query) / (
            1 + self.lmbd_entropy
        )


class ShannonEntroy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -(torch.mean(x * torch.log(x)) + torch.mean((1 - x) * torch.log(1 - x)))
