"""
Multi-task Heads for Turnover & Preference Tasks
================================================
Provides heads for turnover prediction and preference scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TurnoverHeadConfig:
    hidden_dim: int = 128
    dropout: float = 0.3


class TurnoverHead(nn.Module):
    """MLP head for turnover prediction."""

    def __init__(self, input_dim: int, config: TurnoverHeadConfig = TurnoverHeadConfig()) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, employee_emb: torch.Tensor, current_job_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([employee_emb, current_job_emb], dim=-1)
        return self.net(fused).squeeze(-1)


@dataclass
class PreferenceHeadConfig:
    hidden_dim: int = 128
    dropout: float = 0.3
    mode: Literal["concat", "dot"] = "concat"


class PreferencePairwiseHead(nn.Module):
    """
    Pairwise scorer for preference modelling.
    Returns scores for preferred and dispreferred job embeddings.
    """

    def __init__(self, embedding_dim: int, config: PreferenceHeadConfig = PreferenceHeadConfig()) -> None:
        super().__init__()
        self.config = config
        if config.mode == "concat":
            self.scorer = nn.Sequential(
                nn.Linear(embedding_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
        elif config.mode == "dot":
            self.scorer = None
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

    def score(self, employee_emb: torch.Tensor, job_emb: torch.Tensor) -> torch.Tensor:
        if self.config.mode == "dot":
            return (employee_emb * job_emb).sum(dim=-1)
        fused = torch.cat([employee_emb, job_emb], dim=-1)
        return self.scorer(fused).squeeze(-1)

    def forward(
        self,
        employee_emb: torch.Tensor,
        prefer_job_emb: torch.Tensor,
        disprefer_job_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        score_pref = self.score(employee_emb, prefer_job_emb)
        score_dispref = self.score(employee_emb, disprefer_job_emb)
        return score_pref, score_dispref

