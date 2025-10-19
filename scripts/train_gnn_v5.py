"""
Train Heterogeneous GNN (v5)
============================
Multi-task training for turnover classification and job preference ranking.
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
from torch.utils.data import Dataset
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
    PreferenceLossConfig,
    TurnoverLossConfig,
    aggregate_losses,
)
from src.models.layers.categorical_embedding import CategoricalEmbedding  # noqa: F401  # (placeholder for future use)
from src.models.trainer import compute_metrics

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install PyYAML to run this script.") from exc


class PreferenceTripleDataset(Dataset):
    """Stores (employee, prefer_post, disprefer_post) triples."""

    def __init__(self, triples: torch.Tensor) -> None:
        self.triples = triples.long()

    def __len__(self) -> int:
        return self.triples.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.triples[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hetero GNN (v5)")
    parser.add_argument("--config", type=Path, default=Path("configs/hetero/default.yaml"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. training.lr=0.0005 loss.alpha=1.5",
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


def prepare_triples(triples_path: Path, device: torch.device, mask: torch.Tensor) -> torch.Tensor:
    triples = torch.load(triples_path).long()
    if triples.ndim != 2 or triples.size(1) != 3:
        raise ValueError("Preference triples tensor must have shape [N, 3]")
    triples = triples[mask[triples[:, 0]].cpu().numpy()]
    return torch.tensor(triples, dtype=torch.long, device=device)


def threshold_search(labels: torch.Tensor, probs: torch.Tensor, thresholds: np.ndarray) -> Tuple[float, Dict[str, float]]:
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


def compute_preference_metrics(employee_emb: torch.Tensor, post_emb: torch.Tensor, triples: torch.Tensor) -> Dict[str, float]:
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

    data_cfg = config["data"]
    data = load_data(Path(data_cfg["graph_path"]), device)

    scaler_path = Path(data_cfg["scaler_path"])
    normalizer = FeatureNormalizer(scaler_path)
    data["employee"].x = normalizer(data["employee"].x.float())

    triples_all = torch.load(Path(data_cfg["triples_path"])).long().to(device)

    train_mask = data["employee"].train_mask.bool()
    val_mask = data["employee"].val_mask.bool()
    test_mask = data["employee"].test_mask.bool()

    train_triples = triples_all[train_mask[triples_all[:, 0]]]
    val_triples = triples_all[val_mask[triples_all[:, 0]]]
    test_triples = triples_all[test_mask[triples_all[:, 0]]]

    metadata = data.metadata()
    input_dims = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

    model_cfg = HeteroGNNConfig(**config["model"])
    model = HeteroGNN(metadata, input_dims, model_cfg).to(device)

    turnover_cfg = TurnoverHeadConfig(hidden_dim=int(config["model"]["hidden_dim"]))
    turnover_head = TurnoverHead(input_dim=config["model"]["hidden_dim"] * 2, config=turnover_cfg).to(device)

    pref_cfg_dict = config["loss"].get("preference_head", {})
    preference_head = PreferencePairwiseHead(
        embedding_dim=config["model"]["hidden_dim"],
        config=PreferenceHeadConfig(**pref_cfg_dict),
    ).to(device)

    params = list(model.parameters()) + list(turnover_head.parameters()) + list(preference_head.parameters())

    training_cfg = config["training"]
    optimizer = torch.optim.Adam(params, lr=training_cfg["lr"], weight_decay=training_cfg["weight_decay"])

    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "reduce_on_plateau")
    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
            verbose=True,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_cfg["epochs"],
        )
    else:
        scheduler = None

    turnover_loss_cfg = TurnoverLossConfig(**config["loss"]["turnover"])
    preference_loss_cfg = PreferenceLossConfig(**config["loss"]["preference"])
    alpha = float(config["loss"]["alpha"])
    beta = float(config["loss"]["beta"])
    gamma = float(config["loss"].get("gamma", 0.0))

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    save_dir = Path(config.get("logging", {}).get("save_dir", "outputs/hetero")) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_pref_acc": [], "val_pref_acc": [], "lr": []}
    best_state = None
    best_val_loss = float("inf")
    patience = training_cfg["patience"]
    patience_counter = 0

    current_job_idx = data["employee", "assigned_to", "current_job"].edge_index[1].long()
    labels = data["employee"].y.long()

    for epoch in range(1, training_cfg["epochs"] + 1):
        model.train()
        turnover_head.train()
        preference_head.train()

        optimizer.zero_grad()

        embeddings = model(data)
        employee_emb = embeddings["employee"]
        current_job_emb = embeddings["current_job"]
        post_emb = embeddings["post_type"]

        turnover_logits = turnover_head(employee_emb, current_job_emb[current_job_idx])

        train_emp = train_triples[:, 0]
        train_pref = train_triples[:, 1]
        train_disp = train_triples[:, 2]
        pref_scores, disp_scores = preference_head(
            employee_emb[train_emp],
            post_emb[train_pref],
            post_emb[train_disp],
        )

        loss_dict = aggregate_losses(
            turnover_logits[train_mask],
            labels[train_mask],
            pref_scores,
            disp_scores,
            alpha=alpha,
            beta=beta,
            turnover_cfg=turnover_loss_cfg,
            preference_cfg=preference_loss_cfg,
            aux_loss=None,
            gamma=gamma,
        )

        loss = loss_dict["loss_total"]
        loss.backward()

        if training_cfg.get("grad_clip", None):
            torch.nn.utils.clip_grad_norm_(params, training_cfg["grad_clip"])

        optimizer.step()

        train_pref_metrics = compute_preference_metrics(employee_emb, post_emb, train_triples)

        with torch.no_grad():
            model.eval()
            turnover_head.eval()
            preference_head.eval()

            embeddings_val = model(data)
            employee_val = embeddings_val["employee"]
            current_job_val = embeddings_val["current_job"]
            post_val = embeddings_val["post_type"]

            val_logits = turnover_head(employee_val, current_job_val[current_job_idx])
            val_pref_scores, val_disp_scores = preference_head(
                employee_val[val_triples[:, 0]],
                post_val[val_triples[:, 1]],
                post_val[val_triples[:, 2]],
            )

            val_losses = aggregate_losses(
                val_logits[val_mask],
                labels[val_mask],
                val_pref_scores,
                val_disp_scores,
                alpha=alpha,
                beta=beta,
                turnover_cfg=turnover_loss_cfg,
                preference_cfg=preference_loss_cfg,
            )

            val_pref_metrics = compute_preference_metrics(employee_val, post_val, val_triples)

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_losses["loss_total"].item())
            history["train_pref_acc"].append(train_pref_metrics["accuracy"])
            history["val_pref_acc"].append(val_pref_metrics["accuracy"])
            history["lr"].append(optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {loss.item():.4f} | "
                f"Val Loss {val_losses['loss_total'].item():.4f} | "
                f"Val PrefAcc {val_pref_metrics['accuracy']:.4f}"
            )

            if val_losses["loss_total"].item() < best_val_loss:
                best_val_loss = val_losses["loss_total"].item()
                patience_counter = 0
                best_state = {
                    "model": model.state_dict(),
                    "turnover_head": turnover_head.state_dict(),
                    "preference_head": preference_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                }
                torch.save(best_state, save_dir / "best_model.pt")
            else:
                patience_counter += 1

            if scheduler_type == "reduce_on_plateau" and scheduler is not None:
                scheduler.step(val_losses["loss_total"].item())
            elif scheduler_type == "cosine" and scheduler is not None:
                scheduler.step()

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    torch.save(history, save_dir / "training_history.pt")
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        turnover_head.load_state_dict(best_state["turnover_head"])
        preference_head.load_state_dict(best_state["preference_head"])

    model.eval()
    turnover_head.eval()
    preference_head.eval()

    embeddings_final = model(data)
    employee_final = embeddings_final["employee"]
    current_job_final = embeddings_final["current_job"]
    post_final = embeddings_final["post_type"]
    logits_final = turnover_head(employee_final, current_job_final[current_job_idx])
    probs_final = torch.sigmoid(logits_final)

    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold, val_threshold_metrics = threshold_search(
        labels[val_mask],
        probs_final[val_mask],
        thresholds,
    )
    test_metrics = compute_metrics(
        labels[test_mask].cpu().numpy(),
        probs_final[test_mask].detach().cpu().numpy(),
        threshold=best_threshold,
    )

    val_pref_final = compute_preference_metrics(employee_final, post_final, val_triples)
    test_pref_final = compute_preference_metrics(employee_final, post_final, test_triples)

    results = {
        "run_id": run_id,
        "config": config,
        "best_epoch": best_state["epoch"] if best_state else None,
        "val_loss": best_val_loss,
        "best_threshold": best_threshold,
        "val_metrics": val_threshold_metrics,
        "test_metrics": test_metrics,
        "val_pref_metrics": val_pref_final,
        "test_pref_metrics": test_pref_final,
    }

    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Training complete.")
    print(f"  Run ID: {run_id}")
    print(f"  Best Epoch: {results['best_epoch']}")
    print(f"  Val F1 (best threshold {best_threshold:.2f}): {val_threshold_metrics['f1']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Val Pref Accuracy: {val_pref_final['accuracy']:.4f}")
    print(f"  Test Pref Accuracy: {test_pref_final['accuracy']:.4f}")


if __name__ == "__main__":
    main()
