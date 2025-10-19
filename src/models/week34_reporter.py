"""
Week34 Reporter
===============
Generates evaluation reports for the hetero GNN (turnover + preference).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.manifold import TSNE
from torch_geometric.data import HeteroData

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install PyYAML to run this script.") from exc


def compute_classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        auc = 0.0
    else:
        auc = roc_auc_score(labels, probs)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": auc,
        "aupr": average_precision_score(labels, probs),
    }
    return metrics


def pairwise_accuracy(employee_emb: torch.Tensor, post_emb: torch.Tensor, triples: torch.Tensor) -> float:
    if triples.numel() == 0:
        return 0.0
    employee_idx = triples[:, 0]
    prefer_idx = triples[:, 1]
    disprefer_idx = triples[:, 2]
    pref_scores = (employee_emb[employee_idx] * post_emb[prefer_idx]).sum(dim=-1)
    disp_scores = (employee_emb[employee_idx] * post_emb[disprefer_idx]).sum(dim=-1)
    return (pref_scores > disp_scores).float().mean().item()


def compute_topk_metrics(
    employee_emb: torch.Tensor,
    post_emb: torch.Tensor,
    triples: torch.Tensor,
    ks: List[int],
) -> Dict[str, float]:
    if triples.numel() == 0:
        return {f"recall@{k}": 0.0 for k in ks}

    unique_employees = triples[:, 0].unique()
    scores = employee_emb[unique_employees] @ post_emb.T
    metrics = {}

    for k in ks:
        topk = torch.topk(scores, k=min(k, scores.size(1)), dim=-1).indices
        hits = []
        for idx, emp in enumerate(unique_employees):
            prefer_posts = triples[triples[:, 0] == emp][:, 1].unique()
            hits.append(int(any(post.item() in prefer_posts.tolist() for post in topk[idx])))
        metrics[f"recall@{k}"] = float(np.mean(hits))
    return metrics


def compute_ndcg(employee_emb: torch.Tensor, post_emb: torch.Tensor, triples: torch.Tensor, k: int = 10) -> float:
    if triples.numel() == 0:
        return 0.0
    unique_employees = triples[:, 0].unique()
    scores = employee_emb[unique_employees] @ post_emb.T
    ndcgs = []

    for idx, emp in enumerate(unique_employees):
        relevant = triples[triples[:, 0] == emp][:, 1].unique().tolist()
        if not relevant:
            continue
        ranked = torch.topk(scores[idx], k=min(k, scores.size(1))).indices.tolist()
        dcg = 0.0
        for rank, post in enumerate(ranked, 1):
            if post in relevant:
                dcg += 1 / np.log2(rank + 1)
        ideal_dcg = sum(1 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        if ideal_dcg == 0:
            continue
        ndcgs.append(dcg / ideal_dcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    title: str,
) -> None:
    from matplotlib import pyplot as plt

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings.cpu().numpy())
    labels_np = labels.cpu().numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap="coolwarm", alpha=0.7, s=10)
    plt.title(title)
    plt.colorbar(scatter)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week34 Reporter")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing model outputs (results.json, best_model.pt)")
    parser.add_argument("--hetero-data", type=Path, default=Path("data/processed/hetero_graph.pt"))
    parser.add_argument("--triples", type=Path, default=Path("data/processed/preference_triples.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/hetero/default.yaml"))
    parser.add_argument("--baseline-results", type=Path, nargs="*", default=[], help="Optional results.json from GCN v2/v4")
    parser.add_argument("--tsne", action="store_true", help="Generate TSNE plots for employee embeddings")
    return parser.parse_args()


def load_model(run_dir: Path):
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    return state


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    results_json = run_dir / "results.json"
    if not results_json.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data: HeteroData = torch.load(args.hetero_data).to(device)
    triples = torch.load(args.triples).long().to(device)

    state = load_model(run_dir)

    from src.models.hetero_gnn import HeteroGNN, HeteroGNNConfig
    from src.models.multitask_heads import TurnoverHead, PreferencePairwiseHead, TurnoverHeadConfig, PreferenceHeadConfig

    metadata = data.metadata()
    input_dims = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

    model = HeteroGNN(metadata, input_dims, HeteroGNNConfig(**config["model"]))
    model.load_state_dict(state["model"])
    model.to(device).eval()

    turnover_head = TurnoverHead(config["model"]["hidden_dim"] * 2, TurnoverHeadConfig(hidden_dim=config["model"]["hidden_dim"]))
    turnover_head.load_state_dict(state["turnover_head"])
    turnover_head.to(device).eval()

    pref_cfg = PreferenceHeadConfig(**config["loss"].get("preference_head", {}))
    preference_head = PreferencePairwiseHead(config["model"]["hidden_dim"], pref_cfg)
    preference_head.load_state_dict(state["preference_head"])
    preference_head.to(device).eval()

    with torch.no_grad():
        embeddings = model(data)
        employee_emb = embeddings["employee"]
        current_job_emb = embeddings["current_job"]
        post_emb = embeddings["post_type"]

        current_job_idx = data["employee", "assigned_to", "current_job"].edge_index[1]
        logits = turnover_head(employee_emb, current_job_emb[current_job_idx])
        probs = torch.sigmoid(logits)
        labels = data["employee"].y

        train_mask = data["employee"].train_mask.bool()
        val_mask = data["employee"].val_mask.bool()
        test_mask = data["employee"].test_mask.bool()

        thresholds = np.arange(0.05, 0.95, 0.01)
        best_threshold = results.get("best_threshold", 0.5)

        test_metrics = compute_classification_metrics(
            labels[test_mask].cpu().numpy(),
            probs[test_mask].detach().cpu().numpy(),
            threshold=best_threshold,
        )

        print("\n=== Turnover Metrics (Test) ===")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        val_triples = triples[val_mask[triples[:, 0]]]
        test_triples = triples[test_mask[triples[:, 0]]]

        print("\n=== Preference Metrics ===")
        print(f"Val pairwise accuracy: {pairwise_accuracy(employee_emb, post_emb, val_triples):.4f}")
        print(f"Test pairwise accuracy: {pairwise_accuracy(employee_emb, post_emb, test_triples):.4f}")

        for k in [3, 5, 10]:
            val_topk = compute_topk_metrics(employee_emb, post_emb, val_triples, [k])
            test_topk = compute_topk_metrics(employee_emb, post_emb, test_triples, [k])
            print(f"Val recall@{k}: {val_topk[f'recall@{k}']:.4f}")
            print(f"Test recall@{k}: {test_topk[f'recall@{k}']:.4f}")

        print(f"Val NDCG@5: {compute_ndcg(employee_emb, post_emb, val_triples, k=5):.4f}")
        print(f"Test NDCG@5: {compute_ndcg(employee_emb, post_emb, test_triples, k=5):.4f}")

        if args.tsne:
            visualize_embeddings(employee_emb, labels, run_dir / "tsne_employee.png", "Employee Embeddings (Turnover)")

    if args.baseline_results:
        print("\n=== Baseline Comparison ===")
        for path in args.baseline_results:
            path = Path(path)
            if not path.exists():
                print(f"[WARN] Baseline result not found: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                baseline_res = json.load(f)
            name = path.parent.name
            metrics = baseline_res.get("test_metrics", {})
            print(f"[{name}] Test F1: {metrics.get('f1', 0):.4f}, AUROC: {metrics.get('auroc', 0):.4f}, AUPR: {metrics.get('aupr', 0):.4f}")


if __name__ == "__main__":
    main()
