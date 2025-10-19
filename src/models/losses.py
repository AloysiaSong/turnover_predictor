"""
Loss utilities for multi-task training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn.functional as F


@dataclass
class TurnoverLossConfig:
    loss_type: Literal["bce", "focal"] = "bce"
    alpha: float = 0.25
    gamma: float = 2.0
    pos_weight: Optional[float] = None


@dataclass
class PreferenceLossConfig:
    loss_type: Literal["margin", "bpr"] = "margin"
    margin: float = 1.0


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


def compute_preference_loss(
    score_pref: torch.Tensor,
    score_disprefer: torch.Tensor,
    config: PreferenceLossConfig,
) -> torch.Tensor:
    if config.loss_type == "margin":
        target = torch.ones_like(score_pref)
        return F.margin_ranking_loss(score_pref, score_disprefer, target, margin=config.margin)
    if config.loss_type == "bpr":
        return -F.logsigmoid(score_pref - score_disprefer).mean()
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
) -> Dict[str, torch.Tensor]:
    loss_turnover = compute_turnover_loss(turnover_logits, turnover_labels, turnover_cfg)
    loss_preference = compute_preference_loss(score_pref, score_disprefer, preference_cfg)
    total = alpha * loss_turnover + beta * loss_preference
    if aux_loss is not None:
        total = total + gamma * aux_loss
    return {
        "loss_total": total,
        "loss_turnover": loss_turnover.detach(),
        "loss_preference": loss_preference.detach(),
        "loss_aux": aux_loss.detach() if aux_loss is not None else torch.tensor(0.0, device=total.device),
    }
