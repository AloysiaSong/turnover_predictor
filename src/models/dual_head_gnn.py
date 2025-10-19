"""
Dual-Head Heterogeneous GNN
===========================

Solves the Dot vs Concat trade-off by using separate projection heads
for Turnover and Preference tasks.

Key Innovation:
- Shared GNN encoder learns universal node representations
- Turnover head uses non-normalized embeddings (preserves information)
- Preference head uses L2-normalized embeddings (suitable for dot product)

This architecture enables simultaneous optimization of both tasks without
the performance collapse observed in single-head models.

Reference:
    Based on multi-task learning with task-specific projections.
    Inspired by CLIP (Radford et al., 2021) and multi-domain learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from .hetero_gnn import HeteroGNN, HeteroGNNConfig


@dataclass
class DualHeadConfig:
    """Configuration for Dual-Head architecture."""

    # Projection dimensions
    turnover_proj_dim: int = 128
    preference_proj_dim: int = 128

    # Projection settings
    use_projection_dropout: bool = True
    projection_dropout: float = 0.1

    # Normalization
    use_batch_norm: bool = False
    use_layer_norm: bool = True

    # Preference head normalization
    normalize_preference: bool = True  # L2 normalization for dot product


class DualHeadGNN(nn.Module):
    """
    Dual-Head Heterogeneous GNN with separate projections for different tasks.

    Architecture:
        Input → Shared HeteroGNN → [Turnover Projection] → Turnover Head
                                  ↘ [Preference Projection] → Preference Head (normalized)

    Args:
        metadata: Graph metadata (node types, edge types)
        input_dims: Input dimensions for each node type
        gnn_config: Configuration for the shared GNN encoder
        dual_head_config: Configuration for dual-head projections
    """

    def __init__(
        self,
        metadata: Tuple[list, list],
        input_dims: Dict[str, int],
        gnn_config: HeteroGNNConfig,
        dual_head_config: DualHeadConfig = None,
    ) -> None:
        super().__init__()

        self.metadata = metadata
        self.gnn_config = gnn_config
        self.dual_head_config = dual_head_config or DualHeadConfig()

        # Shared GNN encoder
        self.shared_gnn = HeteroGNN(metadata, input_dims, gnn_config)

        # Task-specific projection heads
        self._build_projection_heads()

    def _build_projection_heads(self) -> None:
        """Build separate projection heads for each task."""

        hidden_dim = self.gnn_config.hidden_dim
        config = self.dual_head_config

        # Turnover projection (preserves original embedding space)
        turnover_layers = []
        turnover_layers.append(nn.Linear(hidden_dim, config.turnover_proj_dim))

        if config.use_batch_norm:
            turnover_layers.append(nn.BatchNorm1d(config.turnover_proj_dim))
        elif config.use_layer_norm:
            turnover_layers.append(nn.LayerNorm(config.turnover_proj_dim))

        turnover_layers.append(nn.ReLU())

        if config.use_projection_dropout:
            turnover_layers.append(nn.Dropout(config.projection_dropout))

        self.turnover_proj = nn.Sequential(*turnover_layers)

        # Preference projection (will be L2-normalized)
        preference_layers = []
        preference_layers.append(nn.Linear(hidden_dim, config.preference_proj_dim))

        if config.use_batch_norm:
            preference_layers.append(nn.BatchNorm1d(config.preference_proj_dim))
        elif config.use_layer_norm:
            preference_layers.append(nn.LayerNorm(config.preference_proj_dim))

        preference_layers.append(nn.ReLU())

        if config.use_projection_dropout:
            preference_layers.append(nn.Dropout(config.projection_dropout))

        self.preference_proj = nn.Sequential(*preference_layers)

    def forward(
        self,
        data: HeteroData,
        task: str = "both",
    ) -> Dict[str, Tensor] | Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Forward pass with task-specific projections.

        Args:
            data: Heterogeneous graph data
            task: Which task to compute ("turnover", "preference", or "both")

        Returns:
            If task="both": (turnover_embeddings, preference_embeddings)
            Otherwise: task-specific embeddings dictionary
        """
        # Shared encoding
        shared_embeddings = self.shared_gnn(data)

        if task == "turnover":
            return self._project_for_turnover(shared_embeddings)
        elif task == "preference":
            return self._project_for_preference(shared_embeddings)
        elif task == "both":
            turnover_emb = self._project_for_turnover(shared_embeddings)
            preference_emb = self._project_for_preference(shared_embeddings)
            return turnover_emb, preference_emb
        else:
            raise ValueError(f"Unknown task: {task}")

    def _project_for_turnover(
        self,
        shared_embeddings: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Project shared embeddings for turnover prediction task.

        Returns non-normalized embeddings to preserve information for concat.
        """
        turnover_embeddings = {}
        for node_type, emb in shared_embeddings.items():
            turnover_embeddings[node_type] = self.turnover_proj(emb)
        return turnover_embeddings

    def _project_for_preference(
        self,
        shared_embeddings: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Project shared embeddings for preference ranking task.

        Returns L2-normalized embeddings for dot product scoring.
        """
        preference_embeddings = {}
        for node_type, emb in shared_embeddings.items():
            projected = self.preference_proj(emb)

            # L2 normalization for dot product
            if self.dual_head_config.normalize_preference:
                projected = F.normalize(projected, p=2, dim=-1)

            preference_embeddings[node_type] = projected

        return preference_embeddings

    def get_embeddings_for_tasks(
        self,
        data: HeteroData,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Convenience method to get both task embeddings.

        Returns:
            (turnover_embeddings, preference_embeddings)
        """
        return self.forward(data, task="both")


class DualHeadMultiTaskModel(nn.Module):
    """
    Complete multi-task model with Dual-Head GNN and task heads.

    This is a wrapper that combines:
    - DualHeadGNN (encoder with dual projections)
    - TurnoverHead (classification head)
    - PreferencePairwiseHead (ranking head)
    """

    def __init__(
        self,
        dual_head_gnn: DualHeadGNN,
        turnover_head: nn.Module,
        preference_head: nn.Module,
    ) -> None:
        super().__init__()

        self.dual_head_gnn = dual_head_gnn
        self.turnover_head = turnover_head
        self.preference_head = preference_head

    def forward(
        self,
        data: HeteroData,
        current_job_idx: Tensor,
        employee_ids: Tensor = None,
        preferred_posts: Tensor = None,
        negative_posts: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for both tasks.

        Args:
            data: Graph data
            current_job_idx: Indices of current jobs for employees
            employee_ids: Employee IDs for preference task (optional)
            preferred_posts: Preferred post IDs (optional)
            negative_posts: Negative post IDs (optional)

        Returns:
            (turnover_logits, pref_scores, neg_scores)
        """
        # Get task-specific embeddings
        turnover_emb, preference_emb = self.dual_head_gnn.get_embeddings_for_tasks(data)

        # Turnover prediction
        employee_turnover = turnover_emb["employee"]
        job_turnover = turnover_emb["current_job"]
        turnover_logits = self.turnover_head(employee_turnover, job_turnover[current_job_idx])

        # Preference ranking (if provided)
        if employee_ids is not None and preferred_posts is not None and negative_posts is not None:
            employee_pref = preference_emb["employee"]
            post_pref = preference_emb["post_type"]

            pref_scores, neg_scores = self.preference_head(
                employee_pref[employee_ids],
                post_pref[preferred_posts],
                post_pref[negative_posts],
            )
        else:
            pref_scores = torch.tensor(0.0, device=turnover_logits.device)
            neg_scores = torch.tensor(0.0, device=turnover_logits.device)

        return turnover_logits, pref_scores, neg_scores


# Utility function for creating Dual-Head model
def create_dual_head_model(
    metadata: Tuple[list, list],
    input_dims: Dict[str, int],
    gnn_config: HeteroGNNConfig,
    dual_head_config: DualHeadConfig = None,
) -> DualHeadGNN:
    """
    Factory function to create a Dual-Head GNN model.

    Args:
        metadata: Graph metadata
        input_dims: Input dimensions for each node type
        gnn_config: GNN configuration
        dual_head_config: Dual-head configuration

    Returns:
        Configured DualHeadGNN model
    """
    return DualHeadGNN(
        metadata=metadata,
        input_dims=input_dims,
        gnn_config=gnn_config,
        dual_head_config=dual_head_config,
    )
