"""
Loss utilities for multi-task training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TurnoverLossConfig:
    loss_type: Literal["bce", "focal"] = "bce"
    alpha: float = 0.25
    gamma: float = 2.0
    pos_weight: Optional[float] = None


@dataclass
class PreferenceLossConfig:
    loss_type: Literal["margin", "bpr", "adaptive_margin"] = "margin"
    margin: float = 1.0
    max_margin: float = 3.0
    margin_growth: float = 0.1
    ranking_weight: float = 0.1
    hard_negative_weight: float = 2.0


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    if pos_weight is not None:
        pos_tensor = torch.tensor(pos_weight, device=logits.device)
    else:
        pos_tensor = None
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_tensor, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    if alpha is not None:
        alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)
        focal_weight = focal_weight * alpha_weight
    loss = focal_weight * bce
    return loss.mean()


def compute_turnover_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: TurnoverLossConfig,
) -> torch.Tensor:
    if config.loss_type == "bce":
        pos_weight = None if config.pos_weight is None else torch.tensor(config.pos_weight, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
    if config.loss_type == "focal":
        return focal_loss_with_logits(
            logits,
            labels.float(),
            alpha=config.alpha,
            gamma=config.gamma,
            pos_weight=config.pos_weight,
        )
    raise ValueError(f"Unknown turnover loss type: {config.loss_type}")


class AdaptiveMarginLoss(nn.Module):
    """
    Adaptive Margin Ranking Loss with dynamic margin adjustment.

    Features:
    - Margin grows as model improves (curriculum learning)
    - Hard negatives receive higher weights
    - Ranking regularization for better score calibration

    Args:
        initial_margin: Starting margin value
        max_margin: Maximum margin value
        margin_growth: How much to increase margin when model improves
        ranking_weight: Weight for ranking regularization term
        hard_negative_weight: Multiplier for hard negative samples
    """

    def __init__(
        self,
        initial_margin: float = 1.0,
        max_margin: float = 3.0,
        margin_growth: float = 0.1,
        ranking_weight: float = 0.1,
        hard_negative_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.current_margin = initial_margin
        self.max_margin = max_margin
        self.margin_growth = margin_growth
        self.ranking_weight = ranking_weight
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        score_pref: torch.Tensor,
        score_neg: torch.Tensor,
        is_hard_negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute adaptive margin ranking loss.

        Args:
            score_pref: Scores for preferred items [batch_size]
            score_neg: Scores for negative items [batch_size]
            is_hard_negative: Boolean mask for hard negatives [batch_size]

        Returns:
            Scalar loss value
        """
        # Base margin ranking loss
        target = torch.ones_like(score_pref)
        base_loss = F.margin_ranking_loss(
            score_pref,
            score_neg,
            target,
            margin=self.current_margin,
            reduction='none',
        )

        # Apply higher weights to hard negatives
        if is_hard_negative is not None:
            weights = torch.where(
                is_hard_negative,
                torch.tensor(self.hard_negative_weight, device=base_loss.device),
                torch.tensor(1.0, device=base_loss.device),
            )
            base_loss = base_loss * weights

        # Ranking regularization: encourage proper score separation
        # This term ensures that preferred items get consistently higher scores
        ranking_reg = -torch.log(torch.sigmoid(score_pref - score_neg) + 1e-8)

        # Combine losses
        total_loss = base_loss.mean() + self.ranking_weight * ranking_reg.mean()

        return total_loss

    def update_margin(self, val_pref_acc: float) -> None:
        """
        Dynamically adjust margin based on validation performance.

        Args:
            val_pref_acc: Validation preference accuracy (0-1)
        """
        if val_pref_acc > 0.6:
            # Model is doing well, increase difficulty
            self.current_margin = min(
                self.current_margin + self.margin_growth,
                self.max_margin,
            )
        elif val_pref_acc < 0.52:
            # Model struggling, decrease difficulty
            self.current_margin = max(
                self.current_margin - self.margin_growth,
                0.5,
            )

    def get_current_margin(self) -> float:
        """Get the current margin value."""
        return self.current_margin


def compute_preference_loss(
    score_pref: torch.Tensor,
    score_disprefer: torch.Tensor,
    config: PreferenceLossConfig,
    is_hard_negative: Optional[torch.Tensor] = None,
    adaptive_loss_module: Optional[AdaptiveMarginLoss] = None,
) -> torch.Tensor:
    """
    Compute preference ranking loss.

    Args:
        score_pref: Scores for preferred items
        score_disprefer: Scores for dispreferred items
        config: Loss configuration
        is_hard_negative: Optional mask for hard negatives
        adaptive_loss_module: Optional pre-initialized AdaptiveMarginLoss module

    Returns:
        Scalar loss value
    """
    if config.loss_type == "margin":
        target = torch.ones_like(score_pref)
        return F.margin_ranking_loss(score_pref, score_disprefer, target, margin=config.margin)

    if config.loss_type == "adaptive_margin":
        # Use adaptive margin loss
        if adaptive_loss_module is None:
            # Create temporary module for this call
            loss_fn = AdaptiveMarginLoss(
                initial_margin=config.margin,
                max_margin=config.max_margin,
                margin_growth=config.margin_growth,
                ranking_weight=config.ranking_weight,
                hard_negative_weight=config.hard_negative_weight,
            )
        else:
            loss_fn = adaptive_loss_module

        return loss_fn(score_pref, score_disprefer, is_hard_negative)

    if config.loss_type == "bpr":
        # Bayesian Personalized Ranking loss
        base_loss = -F.logsigmoid(score_pref - score_disprefer)

        # Apply hard negative weighting if available
        if is_hard_negative is not None:
            weights = torch.where(
                is_hard_negative,
                torch.tensor(config.hard_negative_weight, device=base_loss.device),
                torch.tensor(1.0, device=base_loss.device),
            )
            base_loss = base_loss * weights

        return base_loss.mean()

    raise ValueError(f"Unknown preference loss type: {config.loss_type}")


def aggregate_losses(
    turnover_logits: torch.Tensor,
    turnover_labels: torch.Tensor,
    score_pref: torch.Tensor,
    score_disprefer: torch.Tensor,
    alpha: float,
    beta: float,
    turnover_cfg: TurnoverLossConfig,
    preference_cfg: PreferenceLossConfig,
    aux_loss: Optional[torch.Tensor] = None,
    gamma: float = 0.0,
    is_hard_negative: Optional[torch.Tensor] = None,
    adaptive_loss_module: Optional[AdaptiveMarginLoss] = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate multiple task losses with optional hard negative support.

    Args:
        turnover_logits: Logits for turnover prediction
        turnover_labels: Ground truth turnover labels
        score_pref: Scores for preferred items
        score_disprefer: Scores for dispreferred items
        alpha: Weight for turnover loss
        beta: Weight for preference loss
        turnover_cfg: Turnover loss configuration
        preference_cfg: Preference loss configuration
        aux_loss: Optional auxiliary loss term
        gamma: Weight for auxiliary loss
        is_hard_negative: Optional mask for hard negatives
        adaptive_loss_module: Optional adaptive margin loss module

    Returns:
        Dictionary containing total loss and individual loss components
    """
    loss_turnover = compute_turnover_loss(turnover_logits, turnover_labels, turnover_cfg)
    loss_preference = compute_preference_loss(
        score_pref,
        score_disprefer,
        preference_cfg,
        is_hard_negative=is_hard_negative,
        adaptive_loss_module=adaptive_loss_module,
    )
    total = alpha * loss_turnover + beta * loss_preference
    if aux_loss is not None:
        total = total + gamma * aux_loss
    return {
        "loss_total": total,
        "loss_turnover": loss_turnover.detach(),
        "loss_preference": loss_preference.detach(),
        "loss_aux": aux_loss.detach() if aux_loss is not None else torch.tensor(0.0, device=total.device),
    }
