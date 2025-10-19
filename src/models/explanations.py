"""
解释模块
========
提供离职预测与偏好排序的局部解释工具。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def _load_feature_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _node_name(node_type: str, idx: int, meta: Dict[str, object]) -> str:
    if node_type == "current_job":
        names = meta.get("job_alias", [])
        if 0 <= idx < len(names):
            return names[idx]
        if idx == len(names):
            return "unknown"
    if node_type == "post_type":
        names = meta.get("post_type_names", [])
        if 0 <= idx < len(names):
            return names[idx]
    if node_type == "company_size":
        mapping = {v: k for k, v in meta.get("company_size_map", {}).items()}
        return mapping.get(idx, f"size_{idx}")
    if node_type == "company_type":
        mapping = {v: k for k, v in meta.get("company_type_map", {}).items()}
        return mapping.get(idx, f"type_{idx}")
    return f"{node_type}_{idx}"


def compute_feature_contributions(
    model,
    turnover_head,
    data: HeteroData,
    employee_ids: Iterable[int],
    feature_names_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    top_k: int = 5,
) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
    """
    使用输入梯度 * 特征的方法近似每个特征对 logit 的贡献。
    """
    model.eval()
    turnover_head.eval()

    feature_names = _load_feature_names(
        feature_names_path or Path("data/processed/feature_names.txt")
    )

    employee_ids = list(int(i) for i in employee_ids)
    if not employee_ids:
        return {}

    data_local = data.clone()
    if device is not None:
        data_local = data_local.to(device)

    x = data_local["employee"].x.clone().detach().requires_grad_(True)
    data_local["employee"].x = x

    with torch.enable_grad():
        embeddings = model(data_local)
        current_job_idx = data_local["employee", "assigned_to", "current_job"].edge_index[1]
        logits = turnover_head(embeddings["employee"], embeddings["current_job"][current_job_idx])

        contributions: Dict[int, Dict[str, List[Tuple[str, float]]]] = {}
        for emp_id in employee_ids:
            if emp_id >= x.size(0):
                continue
            model.zero_grad()
            turnover_head.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

            logit = logits[emp_id]
            logit.backward(retain_graph=True)
            grad = x.grad[emp_id].detach().cpu()
            feature_vals = x.detach().cpu()[emp_id]
            contrib_vec = (grad * feature_vals).numpy()

            if feature_names and len(feature_names) == len(contrib_vec):
                pairs = list(zip(feature_names, contrib_vec))
            else:
                pairs = [(f"feature_{i}", float(v)) for i, v in enumerate(contrib_vec)]

            top_positive = sorted(pairs, key=lambda p: -p[1])[:top_k]
            top_negative = sorted(pairs, key=lambda p: p[1])[:top_k]
            contributions[emp_id] = {
                "top_positive": [(name, float(score)) for name, score in top_positive],
                "top_negative": [(name, float(score)) for name, score in top_negative],
                "logit": float(logit.detach().cpu().item()),
            }
    return contributions


def extract_attention_weights(
    model,
    data: HeteroData,
    employee_ids: Iterable[int],
    meta_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    top_k: int = 5,
) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
    """
    聚合 employee→(current_job/post_type/company_*) 的注意力权重。
    """
    model.eval()
    data_local = data.clone()
    if device is not None:
        data_local = data_local.to(device)

    with torch.no_grad():
        model(data_local)

    attention = getattr(model, "last_attention", {})
    if not attention:
        return {}

    meta = _load_metadata(meta_path or Path("data/processed/hetero_graph_meta.json"))
    employee_ids = list(int(i) for i in employee_ids)
    results: Dict[int, Dict[str, List[Tuple[str, float]]]] = {emp: {} for emp in employee_ids}

    for key, att in attention.items():
        src, rel, dst = key.split("__")
        if src != "employee":
            continue
        edge_type = (src, rel, dst)
        if edge_type not in data.edge_types:
            continue

        edge_index = data[edge_type].edge_index
        att_scores = att
        if att_scores.dim() == 2:
            att_scores = att_scores.mean(dim=-1)
        att_scores = att_scores.float().cpu()

        for emp_id in employee_ids:
            mask = edge_index[0] == emp_id
            if not mask.any():
                continue
            dest_indices = edge_index[1][mask]
            scores = att_scores[mask]
            aggregate: Dict[int, float] = {}
            for dest, score in zip(dest_indices.tolist(), scores.tolist()):
                aggregate[dest] = aggregate.get(dest, 0.0) + score
            sorted_items = sorted(aggregate.items(), key=lambda kv: -kv[1])[:top_k]
            readable = [
                (_node_name(dst, dest_idx, meta), float(val)) for dest_idx, val in sorted_items
            ]
            relation_key = f"{rel}->{dst}"
            results[emp_id].setdefault(relation_key, readable)

    return {emp: info for emp, info in results.items() if info}


def explain_preference(
    model,
    preference_head,
    data: HeteroData,
    triples: Tensor,
    employee_ids: Iterable[int],
    meta_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    top_k: int = 3,
) -> Dict[int, List[Dict[str, float]]]:
    """
    找出 score_pref - score_dispref 最大的偏好 pair，解释员工倾向。
    """
    model.eval()
    preference_head.eval()

    data_local = data.clone()
    if device is not None:
        data_local = data_local.to(device)

    with torch.no_grad():
        embeddings = model(data_local)
        employee_emb = embeddings["employee"]
        post_emb = embeddings["post_type"]

    meta = _load_metadata(meta_path or Path("data/processed/hetero_graph_meta.json"))
    employee_ids = list(int(i) for i in employee_ids)
    triples_cpu = triples.detach().cpu()

    explanations: Dict[int, List[Dict[str, float]]] = {}
    for emp_id in employee_ids:
        mask = triples_cpu[:, 0] == emp_id
        if not mask.any():
            continue
        emp_triples = triples_cpu[mask]
        emp_tensor = torch.tensor(emp_triples, device=employee_emb.device).long()
        prefer_scores, disprefer_scores = preference_head(
            employee_emb[emp_tensor[:, 0]],
            post_emb[emp_tensor[:, 1]],
            post_emb[emp_tensor[:, 2]],
        )
        diff = (prefer_scores - disprefer_scores).detach()
        top_values, top_idx = torch.topk(diff, k=min(top_k, diff.numel()))
        details = []
        for idx_tensor, margin_tensor in zip(top_idx, top_values):
            idx = int(idx_tensor)
            prefer = int(emp_triples[idx, 1])
            disprefer = int(emp_triples[idx, 2])
            details.append(
                {
                    "prefer_post": _node_name("post_type", prefer, meta),
                    "disprefer_post": _node_name("post_type", disprefer, meta),
                    "score_pref": float(prefer_scores[idx].detach().cpu().item()),
                    "score_disprefer": float(disprefer_scores[idx].detach().cpu().item()),
                    "margin": float(margin_tensor.detach().cpu().item()),
                }
            )
        explanations[emp_id] = details
    return explanations


# TODO: 引入 Permutation Importance / SHAP 等全局解释手段，量化整体特征重要性。
