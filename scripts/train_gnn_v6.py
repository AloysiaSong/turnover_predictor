"""
Train Heterogeneous GNN (v6) - Two-Stage Training with Hard Negatives
=====================================================================

Major improvements over v5:
1. Hard negative mining for better preference learning
2. Two-stage training: pre-train on turnover, then fine-tune on preference
3. Adaptive margin loss with curriculum learning
4. Dynamic task weight balancing

This version aims to significantly improve preference learning performance
while maintaining strong turnover prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.hetero_gnn import HeteroGNN, HeteroGNNConfig
from src.models.layers.feature_normalizer import FeatureNormalizer
from src.models.multitask_heads import (
    PreferenceHeadConfig,
    PreferencePairwiseHead,
    TurnoverHead,
    TurnoverHeadConfig,
)
from src.models.losses import (
    AdaptiveMarginLoss,
    PreferenceLossConfig,
    TurnoverLossConfig,
    aggregate_losses,
    compute_turnover_loss,
)
from src.models.sampling import HardNegativeSampler
from src.models.trainer import compute_metrics

try:
    import yaml
except ImportError as exc:
    raise ImportError("Please install PyYAML to run this script.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hetero GNN v6 (two-stage)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hetero/v6_twostage.yaml"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. training.lr=0.0005",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: Any) -> None:
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        d[keys[-1]] = parsed


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def load_data(graph_path: Path, device: torch.device) -> HeteroData:
    data: HeteroData = torch.load(graph_path)
    return data.to(device)


def threshold_search(
    labels: torch.Tensor,
    probs: torch.Tensor,
    thresholds: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    labels_np = labels.cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    best_threshold = 0.5
    best_metrics = compute_metrics(labels_np, probs_np, threshold=best_threshold)
    best_score = best_metrics["f1"]
    for threshold in thresholds:
        metrics = compute_metrics(labels_np, probs_np, threshold=threshold)
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def compute_preference_metrics(
    employee_emb: torch.Tensor,
    post_emb: torch.Tensor,
    triples: torch.Tensor,
) -> Dict[str, float]:
    if triples.numel() == 0:
        return {"accuracy": 0.0, "margin": 0.0}
    emp = triples[:, 0]
    pref = triples[:, 1]
    disp = triples[:, 2]
    pref_scores = (employee_emb[emp] * post_emb[pref]).sum(dim=-1)
    disp_scores = (employee_emb[emp] * post_emb[disp]).sum(dim=-1)
    accuracy = (pref_scores > disp_scores).float().mean().item()
    margin = (pref_scores - disp_scores).mean().item()
    return {"accuracy": accuracy, "margin": margin}


def stage1_pretrain_turnover(
    model: HeteroGNN,
    turnover_head: TurnoverHead,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    labels: torch.Tensor,
    current_job_idx: torch.Tensor,
    config: Dict[str, Any],
    save_dir: Path,
    device: torch.device,
) -> None:
    """
    Stage 1: Pre-train GNN encoder on Turnover task only.

    This allows the GNN to learn good node representations before
    tackling the more difficult preference learning task.
    """
    print("=" * 70)
    print("STAGE 1: Pre-training GNN on Turnover Classification")
    print("=" * 70)

    stage1_cfg = config["training"]["stage1"]

    # Optimizer for stage 1 (GNN + turnover head only)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(turnover_head.parameters()),
        lr=stage1_cfg["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=True,
    )

    # Loss config
    turnover_loss_cfg = TurnoverLossConfig(**config["loss"]["turnover"])

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, stage1_cfg["epochs"] + 1):
        # Training
        model.train()
        turnover_head.train()

        optimizer.zero_grad()

        embeddings = model(data)
        employee_emb = embeddings["employee"]
        current_job_emb = embeddings["current_job"]

        logits = turnover_head(employee_emb, current_job_emb[current_job_idx])
        loss = compute_turnover_loss(
            logits[train_mask],
            labels[train_mask],
            turnover_loss_cfg,
        )

        loss.backward()

        if config["training"].get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(turnover_head.parameters()),
                config["training"]["grad_clip"],
            )

        optimizer.step()

        # Validation
        model.eval()
        turnover_head.eval()

        with torch.no_grad():
            embeddings_val = model(data)
            employee_val = embeddings_val["employee"]
            job_val = embeddings_val["current_job"]

            val_logits = turnover_head(employee_val, job_val[current_job_idx])
            val_loss = compute_turnover_loss(
                val_logits[val_mask],
                labels[val_mask],
                turnover_loss_cfg,
            )

            # Compute validation metrics
            val_probs = torch.sigmoid(val_logits[val_mask])
            val_metrics = compute_metrics(
                labels[val_mask].cpu().numpy(),
                val_probs.cpu().numpy(),
                threshold=0.5,
            )

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_f1"].append(val_metrics["f1"])

        scheduler.step(val_loss.item())

        if epoch % 10 == 0:
            print(
                f"Stage1 Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0

            torch.save({
                "model": model.state_dict(),
                "turnover_head": turnover_head.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, save_dir / "stage1_best.pt")
        else:
            patience_counter += 1

        if patience_counter >= stage1_cfg["patience"]:
            print(f"Stage 1 early stopping at epoch {epoch}")
            break

    # Save stage 1 history
    with open(save_dir / "stage1_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best model
    checkpoint = torch.load(save_dir / "stage1_best.pt")
    model.load_state_dict(checkpoint["model"])
    turnover_head.load_state_dict(checkpoint["turnover_head"])

    print(f"\nStage 1 complete. Best val loss: {best_val_loss:.4f}")


def stage2_finetune_preference(
    model: HeteroGNN,
    turnover_head: TurnoverHead,
    preference_head: PreferencePairwiseHead,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    labels: torch.Tensor,
    current_job_idx: torch.Tensor,
    train_triples: torch.Tensor,
    val_triples: torch.Tensor,
    config: Dict[str, Any],
    save_dir: Path,
    device: torch.device,
) -> Tuple[HeteroGNN, TurnoverHead, PreferencePairwiseHead]:
    """
    Stage 2: Fine-tune on Preference task with hard negative mining.

    Optionally freeze early GNN layers and use smaller learning rate
    to preserve learned representations while adapting to preference task.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Fine-tuning on Preference Ranking with Hard Negatives")
    print("=" * 70)

    stage2_cfg = config["training"]["stage2"]

    # Optionally freeze early GNN layers
    freeze_layers = stage2_cfg.get("freeze_layers", 0)
    if freeze_layers > 0:
        print(f"Freezing first {freeze_layers} GNN layer(s)")
        for i in range(min(freeze_layers, len(model.gnn_layers))):
            for param in model.gnn_layers[i].parameters():
                param.requires_grad = False

    # Initialize hard negative sampler
    num_posts = data["post_type"].x.size(0)
    sampler = HardNegativeSampler(
        num_items=num_posts,
        hard_ratio=stage2_cfg.get("hard_ratio", 0.7),
        cache_size=stage2_cfg.get("cache_size", 5),
        update_freq=stage2_cfg.get("update_freq", 5),
    )

    # Initialize adaptive margin loss
    pref_cfg = PreferenceLossConfig(**config["loss"]["preference"])
    adaptive_margin = AdaptiveMarginLoss(
        initial_margin=pref_cfg.margin,
        max_margin=pref_cfg.max_margin,
        margin_growth=pref_cfg.margin_growth,
        ranking_weight=pref_cfg.ranking_weight,
        hard_negative_weight=pref_cfg.hard_negative_weight,
    ).to(device)

    # Optimizer (only trainable parameters)
    trainable_params = (
        [p for p in model.parameters() if p.requires_grad] +
        list(turnover_head.parameters()) +
        list(preference_head.parameters()) +
        list(adaptive_margin.parameters())
    )

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=stage2_cfg["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # Maximize preference accuracy
        factor=0.5,
        patience=8,
        verbose=True,
    )

    # Loss configs
    turnover_loss_cfg = TurnoverLossConfig(**config["loss"]["turnover"])

    # Task weights for stage 2 (emphasize preference)
    alpha = stage2_cfg.get("alpha", 0.2)  # Lower weight on turnover
    beta = stage2_cfg.get("beta", 0.8)    # Higher weight on preference

    # Training history
    history = {
        "train_loss": [],
        "val_pref_acc": [],
        "val_turnover_f1": [],
        "margin_value": [],
        "sampler_stats": [],
    }

    best_val_pref_acc = 0.0
    patience_counter = 0

    for epoch in range(1, stage2_cfg["epochs"] + 1):
        # Training
        model.train()
        turnover_head.train()
        preference_head.train()
        adaptive_margin.train()

        optimizer.zero_grad()

        embeddings = model(data)
        employee_emb = embeddings["employee"]
        current_job_emb = embeddings["current_job"]
        post_emb = embeddings["post_type"]

        # Update hard negative cache periodically
        sampler.update_cache(employee_emb, post_emb, train_triples)

        # Sample hard negatives for training
        train_emp = train_triples[:, 0]
        train_pref = train_triples[:, 1]
        train_neg = sampler.sample_negatives(train_emp, train_pref, device)

        # Compute preference scores
        pref_scores, neg_scores = preference_head(
            employee_emb[train_emp],
            post_emb[train_pref],
            post_emb[train_neg],
        )

        # Preference loss with adaptive margin
        loss_pref = adaptive_margin(pref_scores, neg_scores)

        # Turnover loss (keep model grounded)
        turnover_logits = turnover_head(
            employee_emb,
            current_job_emb[current_job_idx],
        )
        loss_turn = compute_turnover_loss(
            turnover_logits[train_mask],
            labels[train_mask],
            turnover_loss_cfg,
        )

        # Combined loss
        total_loss = alpha * loss_turn + beta * loss_pref

        total_loss.backward()

        if config["training"].get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(trainable_params, config["training"]["grad_clip"])

        optimizer.step()

        # Validation
        model.eval()
        turnover_head.eval()
        preference_head.eval()
        adaptive_margin.eval()

        with torch.no_grad():
            embeddings_val = model(data)
            employee_val = embeddings_val["employee"]
            job_val = embeddings_val["current_job"]
            post_val = embeddings_val["post_type"]

            # Preference metrics
            val_pref_metrics = compute_preference_metrics(
                employee_val,
                post_val,
                val_triples,
            )

            # Turnover metrics
            val_logits = turnover_head(employee_val, job_val[current_job_idx])
            val_probs = torch.sigmoid(val_logits[val_mask])
            val_turn_metrics = compute_metrics(
                labels[val_mask].cpu().numpy(),
                val_probs.cpu().numpy(),
                threshold=0.5,
            )

        # Update adaptive margin based on performance
        adaptive_margin.update_margin(val_pref_metrics["accuracy"])

        # Record history
        history["train_loss"].append(total_loss.item())
        history["val_pref_acc"].append(val_pref_metrics["accuracy"])
        history["val_turnover_f1"].append(val_turn_metrics["f1"])
        history["margin_value"].append(adaptive_margin.get_current_margin())
        history["sampler_stats"].append(sampler.get_stats())

        scheduler.step(val_pref_metrics["accuracy"])

        if epoch % 5 == 0:
            print(
                f"Stage2 Epoch {epoch:03d} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Val PrefAcc: {val_pref_metrics['accuracy']:.4f} | "
                f"Val TurnF1: {val_turn_metrics['f1']:.4f} | "
                f"Margin: {adaptive_margin.get_current_margin():.2f}"
            )

        # Save best model
        if val_pref_metrics["accuracy"] > best_val_pref_acc:
            best_val_pref_acc = val_pref_metrics["accuracy"]
            patience_counter = 0

            torch.save({
                "model": model.state_dict(),
                "turnover_head": turnover_head.state_dict(),
                "preference_head": preference_head.state_dict(),
                "adaptive_margin": adaptive_margin.state_dict(),
                "epoch": epoch,
                "val_pref_acc": best_val_pref_acc,
            }, save_dir / "stage2_best.pt")
        else:
            patience_counter += 1

        if patience_counter >= stage2_cfg["patience"]:
            print(f"Stage 2 early stopping at epoch {epoch}")
            break

    # Save stage 2 history
    with open(save_dir / "stage2_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best model
    checkpoint = torch.load(save_dir / "stage2_best.pt")
    model.load_state_dict(checkpoint["model"])
    turnover_head.load_state_dict(checkpoint["turnover_head"])
    preference_head.load_state_dict(checkpoint["preference_head"])

    print(f"\nStage 2 complete. Best val pref acc: {best_val_pref_acc:.4f}")

    return model, turnover_head, preference_head


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.override)

    set_seed(int(config.get("seed", 42)))
    device = select_device(config.get("device", "auto"))

    print(f"Using device: {device}")

    # Load data
    data_cfg = config["data"]
    data = load_data(Path(data_cfg["graph_path"]), device)

    # Normalize features
    scaler_path = Path(data_cfg["scaler_path"])
    normalizer = FeatureNormalizer(scaler_path)
    data["employee"].x = normalizer(data["employee"].x.float())

    # Load preference triples
    triples_all = torch.load(Path(data_cfg["triples_path"])).long().to(device)

    # Split masks
    train_mask = data["employee"].train_mask.bool()
    val_mask = data["employee"].val_mask.bool()
    test_mask = data["employee"].test_mask.bool()

    # Filter triples by mask
    train_triples = triples_all[train_mask[triples_all[:, 0]]]
    val_triples = triples_all[val_mask[triples_all[:, 0]]]
    test_triples = triples_all[test_mask[triples_all[:, 0]]]

    print(f"Train triples: {train_triples.size(0)}")
    print(f"Val triples: {val_triples.size(0)}")
    print(f"Test triples: {test_triples.size(0)}")

    # Build model
    metadata = data.metadata()
    input_dims = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

    model_cfg = HeteroGNNConfig(**config["model"])
    model = HeteroGNN(metadata, input_dims, model_cfg).to(device)

    turnover_cfg = TurnoverHeadConfig(hidden_dim=int(config["model"]["hidden_dim"]))
    turnover_head = TurnoverHead(
        input_dim=config["model"]["hidden_dim"] * 2,
        config=turnover_cfg,
    ).to(device)

    pref_head_cfg = PreferenceHeadConfig(**config["loss"].get("preference_head", {}))
    preference_head = PreferencePairwiseHead(
        embedding_dim=config["model"]["hidden_dim"],
        config=pref_head_cfg,
    ).to(device)

    # Setup directories
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    save_dir = Path(config.get("logging", {}).get("save_dir", "outputs/hetero_v6")) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Get edge info
    current_job_idx = data["employee", "assigned_to", "current_job"].edge_index[1].long()
    labels = data["employee"].y.long()

    # ========== Two-Stage Training ==========

    # Stage 1: Pre-train on turnover
    stage1_pretrain_turnover(
        model,
        turnover_head,
        data,
        train_mask,
        val_mask,
        labels,
        current_job_idx,
        config,
        save_dir,
        device,
    )

    # Stage 2: Fine-tune on preference with hard negatives
    model, turnover_head, preference_head = stage2_finetune_preference(
        model,
        turnover_head,
        preference_head,
        data,
        train_mask,
        val_mask,
        labels,
        current_job_idx,
        train_triples,
        val_triples,
        config,
        save_dir,
        device,
    )

    # ========== Final Evaluation ==========

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    model.eval()
    turnover_head.eval()
    preference_head.eval()

    with torch.no_grad():
        embeddings_final = model(data)
        employee_final = embeddings_final["employee"]
        current_job_final = embeddings_final["current_job"]
        post_final = embeddings_final["post_type"]

        logits_final = turnover_head(
            employee_final,
            current_job_final[current_job_idx],
        )
        probs_final = torch.sigmoid(logits_final)

        # Threshold search on validation set
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_threshold, val_metrics = threshold_search(
            labels[val_mask],
            probs_final[val_mask],
            thresholds,
        )

        # Test metrics with best threshold
        test_metrics = compute_metrics(
            labels[test_mask].cpu().numpy(),
            probs_final[test_mask].detach().cpu().numpy(),
            threshold=best_threshold,
        )

        # Preference metrics
        val_pref_metrics = compute_preference_metrics(
            employee_final,
            post_final,
            val_triples,
        )
        test_pref_metrics = compute_preference_metrics(
            employee_final,
            post_final,
            test_triples,
        )

    # Compile results
    results = {
        "run_id": run_id,
        "config": config,
        "best_threshold": best_threshold,
        "val_turnover_metrics": val_metrics,
        "test_turnover_metrics": test_metrics,
        "val_preference_metrics": val_pref_metrics,
        "test_preference_metrics": test_pref_metrics,
    }

    # Save results
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"\nTurnover Prediction (Test):")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  AUPR:      {test_metrics['aupr']:.4f}")
    print(f"\nPreference Ranking (Test):")
    print(f"  Pairwise Accuracy: {test_pref_metrics['accuracy']:.4f}")
    print(f"  Margin:            {test_pref_metrics['margin']:.4f}")
    print(f"\nSaved to: {save_dir}")


if __name__ == "__main__":
    main()
