"""
Categorical Embedding Utilities
===============================
Simple learnable embeddings for categorical node types.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CategoricalEmbedding(nn.Module):
    """
    Learnable embedding table with Xavier initialization.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.embedding.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].zero_()
        self.freeze = freeze

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.embedding(indices)
        return self.embedding(indices)

