"""
Train Dual-Head GNN - Best of Both Worlds
=========================================

Implements the Dual-Head architecture to simultaneously optimize
Turnover and Preference tasks without performance trade-off.

Key Features:
- Separate projections for each task
- L2 normalization only for preference embeddings
- Balanced multi-task learning
- Hard negative mining
- Two-stage training strategy
"""

from __future__ import annotations

import argparse
import json
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

from src.models.dual_head_gnn import (
    DualHeadGNN,
    DualHeadConfig,
)
from src.models.hetero_gnn import HeteroGNNConfig
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
    compute_turnover_loss,
)
from src.models.sampling import HardNegativeSampler
from src.models.trainer import compute_metrics

try:
    import yaml
except ImportError as exc:
    raise ImportError("Please install PyYAML to run this script.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dual-Head GNN")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hetero/dual_head.yaml"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values",
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.override)

    set_seed(int(config.get("seed", 42)))
    device = select_device(config.get("device", "auto"))

    print(f"Using device: {device}")
    print("=" * 70)
    print("DUAL-HEAD GNN TRAINING")
    print("=" * 70)

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

    # Build Dual-Head model
    metadata = data.metadata()
    input_dims = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

    gnn_config = HeteroGNNConfig(**config["model"]["gnn"])
    dual_head_config = DualHeadConfig(**config["model"]["dual_head"])

    model = DualHeadGNN(metadata, input_dims, gnn_config, dual_head_config).to(device)

    print(f"\nModel architecture:")
    print(f"  Shared GNN: {gnn_config.hidden_dim}D, {gnn_config.num_layers} layers")
    print(f"  Turnover projection: {dual_head_config.turnover_proj_dim}D (non-normalized)")
    print(f"  Preference projection: {dual_head_config.preference_proj_dim}D (L2-normalized)")

    # Build task heads
    turnover_cfg = TurnoverHeadConfig(hidden_dim=config["model"]["gnn"]["hidden_dim"])
    turnover_head = TurnoverHead(
        input_dim=dual_head_config.turnover_proj_dim * 2,
        config=turnover_cfg,
    ).to(device)

    pref_head_cfg = PreferenceHeadConfig(mode="dot")  # Always use dot for preference
    preference_head = PreferencePairwiseHead(
        embedding_dim=dual_head_config.preference_proj_dim,
        config=pref_head_cfg,
    ).to(device)

    # Optimizer
    training_cfg = config["training"]
    params = (
        list(model.parameters()) +
        list(turnover_head.parameters()) +
        list(preference_head.parameters())
    )
    optimizer = torch.optim.Adam(
        params,
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=True,
    )

    # Hard negative sampler
    num_posts = data["post_type"].x.size(0)
    sampler = HardNegativeSampler(
        num_items=num_posts,
        hard_ratio=training_cfg.get("hard_ratio", 0.85),
        cache_size=training_cfg.get("cache_size", 10),
        update_freq=training_cfg.get("update_freq", 5),
    )

    # Adaptive margin loss
    pref_cfg = PreferenceLossConfig(**config["loss"]["preference"])
    adaptive_margin = AdaptiveMarginLoss(
        initial_margin=pref_cfg.margin,
        max_margin=pref_cfg.max_margin,
        margin_growth=pref_cfg.margin_growth,
        ranking_weight=pref_cfg.ranking_weight,
        hard_negative_weight=pref_cfg.hard_negative_weight,
    ).to(device)

    # Loss configs
    turnover_loss_cfg = TurnoverLossConfig(**config["loss"]["turnover"])

    # Task weights
    alpha = float(config["loss"]["alpha"])
    beta = float(config["loss"]["beta"])

    # Setup directories
    run_id = args.run_id or datetime.now().strftime("dual_head_%Y%m%d_%H%M%S")
    save_dir = Path(config.get("logging", {}).get("save_dir", "outputs/dual_head")) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Get edge info
    current_job_idx = data["employee", "assigned_to", "current_job"].edge_index[1].long()
    labels = data["employee"].y.long()

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_turnover_f1": [],
        "val_pref_acc": [],
        "margin": [],
    }

    best_val_score = 0.0  # Harmonic mean of F1 and Pref Acc
    patience_counter = 0

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for epoch in range(1, training_cfg["epochs"] + 1):
        # Training
        model.train()
        turnover_head.train()
        preference_head.train()
        adaptive_margin.train()

        optimizer.zero_grad()

        # Forward pass with dual heads
        turnover_emb, preference_emb = model.get_embeddings_for_tasks(data)

        # Update hard negative cache
        sampler.update_cache(
            preference_emb["employee"],
            preference_emb["post_type"],
            train_triples,
        )

        # Sample hard negatives
        train_emp = train_triples[:, 0]
        train_pref = train_triples[:, 1]
        train_neg = sampler.sample_negatives(train_emp, train_pref, device)

        # Turnover task
        turnover_logits = turnover_head(
            turnover_emb["employee"],
            turnover_emb["current_job"][current_job_idx],
        )
        loss_turnover = compute_turnover_loss(
            turnover_logits[train_mask],
            labels[train_mask],
            turnover_loss_cfg,
        )

        # Preference task
        pref_scores, neg_scores = preference_head(
            preference_emb["employee"][train_emp],
            preference_emb["post_type"][train_pref],
            preference_emb["post_type"][train_neg],
        )
        loss_preference = adaptive_margin(pref_scores, neg_scores)

        # Combined loss
        total_loss = alpha * loss_turnover + beta * loss_preference

        total_loss.backward()

        if training_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(params, training_cfg["grad_clip"])

        optimizer.step()

        # Validation
        model.eval()
        turnover_head.eval()
        preference_head.eval()
        adaptive_margin.eval()

        with torch.no_grad():
            turnover_val, preference_val = model.get_embeddings_for_tasks(data)

            # Turnover metrics
            val_logits = turnover_head(
                turnover_val["employee"],
                turnover_val["current_job"][current_job_idx],
            )
            val_loss_turn = compute_turnover_loss(
                val_logits[val_mask],
                labels[val_mask],
                turnover_loss_cfg,
            )

            val_probs = torch.sigmoid(val_logits[val_mask])
            val_turn_metrics = compute_metrics(
                labels[val_mask].cpu().numpy(),
                val_probs.cpu().numpy(),
                threshold=0.5,
            )

            # Preference metrics
            val_pref_metrics = compute_preference_metrics(
                preference_val["employee"],
                preference_val["post_type"],
                val_triples,
            )

            # Combined validation loss
            val_pref_scores, val_neg_scores = preference_head(
                preference_val["employee"][val_triples[:, 0]],
                preference_val["post_type"][val_triples[:, 1]],
                preference_val["post_type"][val_triples[:, 2]],
            )
            val_loss_pref = adaptive_margin(val_pref_scores, val_neg_scores)
            val_loss_total = alpha * val_loss_turn + beta * val_loss_pref

        # Update margin
        adaptive_margin.update_margin(val_pref_metrics["accuracy"])

        # Record history
        history["train_loss"].append(total_loss.item())
        history["val_loss"].append(val_loss_total.item())
        history["val_turnover_f1"].append(val_turn_metrics["f1"])
        history["val_pref_acc"].append(val_pref_metrics["accuracy"])
        history["margin"].append(adaptive_margin.get_current_margin())

        scheduler.step(val_loss_total.item())

        # Compute combined score (prefer sum to harmonic mean to avoid 0)
        f1 = val_turn_metrics["f1"]
        pref_acc = val_pref_metrics["accuracy"]

        # Use weighted sum instead of harmonic mean (avoids 0 problem)
        combined_score = 0.5 * f1 + 0.5 * pref_acc

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Val F1: {f1:.4f} | "
                f"Val PrefAcc: {pref_acc:.4f} | "
                f"Score: {combined_score:.4f} | "
                f"Margin: {adaptive_margin.get_current_margin():.2f}"
            )

        # Save best model (based on combined score)
        if combined_score > best_val_score:
            best_val_score = combined_score
            patience_counter = 0

            torch.save({
                "model": model.state_dict(),
                "turnover_head": turnover_head.state_dict(),
                "preference_head": preference_head.state_dict(),
                "adaptive_margin": adaptive_margin.state_dict(),
                "epoch": epoch,
                "combined_score": combined_score,
                "f1": f1,
                "pref_acc": pref_acc,
            }, save_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= training_cfg["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best model
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model"])
    turnover_head.load_state_dict(checkpoint["turnover_head"])
    preference_head.load_state_dict(checkpoint["preference_head"])

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    model.eval()
    turnover_head.eval()
    preference_head.eval()

    with torch.no_grad():
        turnover_final, preference_final = model.get_embeddings_for_tasks(data)

        # Turnover prediction
        logits_final = turnover_head(
            turnover_final["employee"],
            turnover_final["current_job"][current_job_idx],
        )
        probs_final = torch.sigmoid(logits_final)

        # Threshold search
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_threshold, val_metrics = threshold_search(
            labels[val_mask],
            probs_final[val_mask],
            thresholds,
        )

        test_metrics = compute_metrics(
            labels[test_mask].cpu().numpy(),
            probs_final[test_mask].detach().cpu().numpy(),
            threshold=best_threshold,
        )

        # Preference metrics
        val_pref_metrics = compute_preference_metrics(
            preference_final["employee"],
            preference_final["post_type"],
            val_triples,
        )
        test_pref_metrics = compute_preference_metrics(
            preference_final["employee"],
            preference_final["post_type"],
            test_triples,
        )

    # Compile results
    results = {
        "run_id": run_id,
        "config": config,
        "best_epoch": checkpoint["epoch"],
        "best_combined_score": checkpoint["combined_score"],
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
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Best Combined Score: {checkpoint['combined_score']:.4f}")
    print(f"  (F1: {checkpoint['f1']:.4f}, Pref Acc: {checkpoint['pref_acc']:.4f})")
    print(f"\nTurnover Prediction (Test):")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  AUPR:      {test_metrics['aupr']:.4f}")
    print(f"\nPreference Ranking (Test):")
    print(f"  Pairwise Accuracy: {test_pref_metrics['accuracy']:.4f}")
    print(f"  Margin:            {test_pref_metrics['margin']:.4f}")
    print(f"\nSaved to: {save_dir}")


if __name__ == "__main__":
    main()
