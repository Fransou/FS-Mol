"""
Utils functions for SimpleShot testing
"""
import sys

from dataclasses import dataclass, field

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

import numpy as np
import torch
import torch.nn as nn

from pyprojroot import here as project_root
from tqdm import tqdm as tqdm

sys.path.insert(0, str(project_root()))

from fs_mol.models.simpleshot import SimpleshotNet, SimpleShotConfig
from fs_mol.data import FSMolTaskSample
from fs_mol.utils.metrics import BinaryEvalMetrics, compute_binary_task_metrics


def get_df_embeddings_stored(df, full_embeddings):
    df_embeddings = full_embeddings[full_embeddings.smiles.isin(df.smiles)]
    df_embeddings = df_embeddings.join(
        df[["smiles", "y"]].set_index("smiles"), on="smiles", how="inner"
    )
    return df_embeddings


def prepare_X_y(df_embeddings_support, df_embeddings_query):
    embeddings_keys = [
        k for k in df_embeddings_support.keys() if k.startswith("embedding_")
    ]
    X_support = np.array(df_embeddings_support[embeddings_keys], dtype=np.float32)
    X_query = np.array(df_embeddings_query[embeddings_keys], dtype=np.float32)

    return X_support, X_query


def preprocess_embeddings(X_support, X_query, center_data=True, normalize_norm=True):
    if center_data:
        mean = np.concatenate([X_support, X_query]).mean(axis=0)
        X_support = X_support - mean
        X_query = X_query - mean

    if normalize_norm:
        X_support = X_support / np.linalg.norm(X_support, axis=1, keepdims=True)
        X_query = X_query / np.linalg.norm(X_query, axis=1, keepdims=True)

    return X_support, X_query

class SimpleShotTrainer:
    def __init__(self, config: SimpleShotConfig):
        self.config = config

    def __call__(self, X_support, X_query, y_support):
        X_support, X_query = preprocess_embeddings(
            X_support,
            X_query,
            center_data=self.config.center_data,
            normalize_norm=self.config.normalize_norm,
        )
        proto_pos = X_support[y_support == 1].mean(axis=0)
        proto_neg = X_support[y_support == 0].mean(axis=0)

        model = SimpleshotNet([proto_pos, proto_neg], self.config)
        optimizer = torch.optim.Adam(model.parameters(), self.config.learning_rate)

        for i in range(self.config.epochs):
            optimizer.zero_grad()
            loss = model.get_loss(
                torch.tensor(X_support, dtype=torch.float32),
                torch.tensor(y_support, dtype=torch.float32),
                torch.tensor(X_query, dtype=torch.float32),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad_norm)
            optimizer.step()
        return model


def test_model_fn(
    task_sample: FSMolTaskSample,
    temp_out_folder: str,
    seed: int,
    descriptor: str = "ecfp",
    model_config: SimpleShotConfig = SimpleShotConfig(),
    p_bar: Optional[tqdm] = None,
):
    train_data = task_sample.train_samples
    test_data = task_sample.test_samples

    df_support = pd.DataFrame(
        {
            "smiles": [s.smiles for s in train_data],
            "y": [int(s.bool_label) for s in train_data],
        }
    )
    df_query = pd.DataFrame(
        {
            "smiles": [s.smiles for s in test_data],
            "y": [int(s.bool_label) for s in test_data],
        }
    )
    if descriptor.endswith(".csv"):
        all_embeddings = pd.read_csv(descriptor)
        df_embeddings_support = get_df_embeddings_stored(df_support, all_embeddings)
        df_embeddings_query = get_df_embeddings_stored(df_query, all_embeddings)
    elif descriptor == "ecfp":
        ecfp_support = np.array([s.get_fingerprint() for s in train_data])
        df_ecfp_support = pd.DataFrame(
            ecfp_support,
            columns=[f"embedding_{i}" for i in range(ecfp_support.shape[1])],
        )

        ecfp_query = np.array([s.get_fingerprint() for s in test_data])
        df_ecfp_query = pd.DataFrame(
            ecfp_query,
            columns=[f"embedding_{i}" for i in range(ecfp_query.shape[1])],
        )

        df_embeddings_support = pd.concat([df_support, df_ecfp_support], axis=1)
        df_embeddings_query = pd.concat([df_query, df_ecfp_query], axis=1)

    else:
        raise ValueError(f"Unknown descriptor {descriptor}")

    X_support, X_query = prepare_X_y(df_embeddings_support, df_embeddings_query)
    y_support = df_embeddings_support.y.to_numpy()
    y_query = df_embeddings_query.y.to_numpy()

    trainer = SimpleShotTrainer(model_config)
    model = trainer(X_support, X_query, y_support)
    y_pred = model.predict(torch.tensor(X_query, dtype=torch.float32)).detach().numpy()

    test_metrics = compute_binary_task_metrics(y_pred, y_query)
    if p_bar is not None:
        p_bar.update(1)
    return test_metrics


@dataclass(frozen=True)
class SimpleShotHPOConfig:
    learning_rate: List[float] = field(default_factory =[1e-3])
    epochs: List[int] = field(default_factory =[10])
    clip_grad_norm: List[float] = field(default_factory =[1.0])
    center_data: List[bool] = field(default_factory =[True, False])
    normalize_norm: List[bool] = field(default_factory =[True, False])
