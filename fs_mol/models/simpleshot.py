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


class SimpleshotNet(nn.Module):
    def __init__(self, input_prototypes, config: SimpleShotConfig):
        super().__init__()
        self.config = config
        self.prototype_pos = nn.Parameter(
            torch.tensor(input_prototypes[0]), requires_grad=True
        )
        self.prototype_neg = nn.Parameter(
            torch.tensor(input_prototypes[1]), requires_grad=True
        )

        self.bias_pos = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.bias_neg = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.lmbd_entropy = config.lmbd_entropy
        self.temperature = config.temperature
        self.bce = nn.BCELoss()
        self.entropy = ShannonEntroy()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        x_pos = self.temperature * (self.cos(x, self.prototype_pos))
        x_neg = self.temperature * (self.cos(x, self.prototype_neg))
        x = torch.sigmoid(x_pos) / (torch.sigmoid(x_pos) + torch.sigmoid(x_neg))
        return x

    def predict(self, x):
        return self.forward(x)

    def get_loss(self, x_support, y_support, x_query):
        y_support_pred = self.forward(x_support)
        cross_entopy_support = self.bce(y_support_pred, y_support)
        y_query_pred = self.forward(x_query)
        entropy_query = self.entropy(y_query_pred)

        return cross_entopy_support + self.lmbd_entropy * entropy_query


class ShannonEntroy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -(torch.mean(x * torch.log(x)) + torch.mean((1 - x) * torch.log(1 - x)))
