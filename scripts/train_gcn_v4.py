"""
GCN 训练脚本 v4
================
目标:
1. 使用GCN编码员工节点特征
2. 将员工embedding与当前岗位embedding融合
3. 通过两层MLP完成离职预测
4. 默认使用带pos_weight的BCEWithLogitsLoss，可选Focal Loss
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

# 确保可以导入 src 模块
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gcn import drop_edges
from src.models.trainer import compute_metrics


def set_seed(seed: int):
    """统一设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """Sigmoid版本的Focal Loss，兼容pos_weight"""

    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N] raw logits
            targets: [N] binary labels (float)
        """
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_t
        loss = bce * focal_weight
        return loss.mean()


class EmployeeGCNEncoder(nn.Module):
    """返回员工embedding的GCN编码器"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,  # 增大隐藏层维度
        num_layers: int = 3,         # 增加层数
        dropout: float = 0.6,       # 增大dropout
        edge_dropout: float = 0.2,  # 增加边dropout
        feature_dropout: float = 0.3
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.feat_dropout = nn.Dropout(feature_dropout) if feature_dropout > 0 else nn.Identity()

        from torch_geometric.nn import GCNConv  # lazy import to avoid global dependency issues

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    @property
    def out_channels(self) -> int:
        return self.hidden_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.feat_dropout(x)
        for i, conv in enumerate(self.convs):
            if self.training and self.edge_dropout > 0.0:
                edge_index = drop_edges(edge_index, self.edge_dropout, x.size(0), training=True)
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EmployeeJobFusionModel(nn.Module):
    """
    组合模型:
    1. 使用GCN获取员工embedding
    2. 查表获得当前岗位embedding
    3. 拼接后送入双层MLP输出离职logit
    """

    def __init__(
        self,
        encoder: EmployeeGCNEncoder,
        num_jobs: int,
        job_embed_dim: int = 32,
        mlp_hidden_dim: int = 128,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.job_embedding = nn.Embedding(num_jobs, job_embed_dim)

        fused_dim = encoder.out_channels + job_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        job_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        employee_emb = self.encoder(x, edge_index)
        job_emb = self.job_embedding(job_indices)
        fused = torch.cat([employee_emb, job_emb], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, employee_emb, job_emb


def detect_job_column(df: pd.DataFrame) -> Optional[str]:
    """
    简单启发式: 查找包含关键字的列名
    优先级: '现岗位' > '当前岗位' > '岗位类型' > 'Q3'/'Q4'等
    """
    candidates = []
    keywords = ["现岗位", "当前岗位", "岗位类型", "岗位类别"]
    for kw in keywords:
        matches = [col for col in df.columns if kw in col]
        candidates.extend(matches)

    if candidates:
        return candidates[0]

    fallback = [col for col in df.columns if col.startswith(("Q2", "Q3", "Q4")) and "_" not in col]
    return fallback[0] if fallback else None


def _normalize_multiselect_value(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s == "":
        return 0
    positives = {"1", "是", "有", "yes", "Yes", "TRUE", "True", "true", "勾选", "选择", "on"}
    negatives = {"0", "否", "无", "no", "No", "FALSE", "False", "false"}
    if s in positives:
        return 1
    if s in negatives:
        return 0
    # 新增: 处理多选字段的多个值（如"1,2,3"）
    if "," in s:
        parts = [part.strip() for part in s.split(",")]
        if any(part in positives for part in parts):
            return 1
        if all(part in negatives for part in parts if part):
            return 0
    try:
        return 1 if float(s) > 0 else 0
    except ValueError:
        return 0


def _is_multiselect_job_col(col_name: str) -> bool:
    text = str(col_name)
    normalized = (
        text.replace("\ufeff", "")
        .replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
        .strip()
    )
    if ("岗位属于" in normalized or "岗位類" in normalized or "岗位类" in normalized) and (
        "多选" in normalized or "多選" in normalized or "可多选" in normalized or "-" in normalized
    ):
        return True
    return False


def _load_multiselect_jobs(
    df: pd.DataFrame,
    job_cols: list,
    raw_header_map: Optional[Dict[str, str]] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    option_names = []
    for col in job_cols:
        source_name = raw_header_map.get(col, col) if raw_header_map else col
        if "-" in source_name:
            option_names.append(source_name.split("-", 1)[-1].strip())
        else:
            option_names.append(str(source_name).strip())

    values = df[job_cols].applymap(_normalize_multiselect_value).values
    num_rows = values.shape[0]
    job_indices = np.zeros(num_rows, dtype=np.int64)

    for idx, row in enumerate(values):
        if row.sum() == 0:
            job_indices[idx] = len(option_names)  # unknown bucket
        else:
            job_indices[idx] = int(np.argmax(row))

    mapping = {name: i for i, name in enumerate(option_names)}
    mapping["未标注"] = len(option_names)

    return job_indices, mapping


def load_job_indices(
    raw_csv: Path,
    num_nodes: int,
    job_column: Optional[str] = None,
    encoding: str = "gbk",
    skiprows: int = 1,
) -> Tuple[torch.LongTensor, Dict[str, int]]:
    """
    从原始CSV中提取当前岗位索引
    """
    df = pd.read_csv(raw_csv, encoding=encoding, skiprows=skiprows)
    if not df.empty and df.iloc[0].astype(str).tolist() == df.columns.astype(str).tolist():
        df = df.iloc[1:].reset_index(drop=True)
    if len(df) < num_nodes:
        raise ValueError(f"原始CSV样本数({len(df)})少于图节点数({num_nodes})")
    if len(df) > num_nodes:
        df = df.iloc[:num_nodes].copy()

    # 优先使用用户指定列
    if job_column is not None:
        column = job_column
        if column not in df.columns:
            raise ValueError(f"指定的岗位列 {column} 不存在于CSV中")
        series = df[column].fillna("未知岗位")
        if series.dtype.kind in {"i", "u"}:
            codes = series.astype(int).values
            unique_vals = np.unique(codes)
            mapping = {str(val): idx for idx, val in enumerate(sorted(unique_vals))}
            remapped = np.vectorize(lambda v: mapping[str(v)])(codes)
        else:
            categories = series.astype("category")
            remapped = categories.cat.codes.values
            mapping = {
                str(cat): int(code)
                for cat, code in zip(categories.cat.categories, range(len(categories.cat.categories)))
            }
        remapped = remapped.astype(np.int64)
        if remapped.min() < 0:
            raise ValueError("岗位编码存在缺失，无法转换，请检查原始CSV")
        return torch.from_numpy(remapped), mapping

    # 读取原始首行列名（用于友好显示）
    raw_header_map: Dict[str, str] = {}
    try:
        import csv

        with open(raw_csv, "r", encoding=encoding) as f:
            reader = csv.reader(f)
            original_header = next(reader)
        if len(original_header) == len(df.columns):
            raw_header_map = dict(zip(df.columns, original_header))
    except Exception:
        raw_header_map = {}

    # 调试信息：展示部分列名
    sample_cols = list(df.columns[:10])
    print("\n   🔍 CSV列名样例:")
    for col in sample_cols:
        print(f"      • {repr(col)}")
    
    job_keyword_cols = [col for col in df.columns if "岗位" in str(col)]
    if job_keyword_cols:
        print(f"\n   🔍 含“岗位”的列数量: {len(job_keyword_cols)}")
        for col in job_keyword_cols[:10]:
            print(f"      • {repr(col)}")
        if len(job_keyword_cols) > 10:
            print(f"      ... 共 {len(job_keyword_cols)} 列")
    else:
        print("\n   🔍 未发现包含“岗位”的列")

    # 优先使用 Q5 多选列（岗位类别）
    q5_cols = [col for col in df.columns if str(col).startswith("Q5_")]
    if q5_cols:
        q5_cols = list(dict.fromkeys(q5_cols))
        print(f"\n   🧩 检测到 Q5 多选列（岗位类别）: {len(q5_cols)} 个")
        for col in q5_cols[:10]:
            display_name = raw_header_map.get(col, col)
            print(f"      - {col} ({display_name})")
        job_indices, mapping = _load_multiselect_jobs(df, q5_cols, raw_header_map)
        return torch.from_numpy(job_indices), mapping

    # 其次尝试多选列
    multiselect_cols = [col for col in df.columns if _is_multiselect_job_col(col)]
    print(f"\n   🔍 多选列匹配数量: {len(multiselect_cols)}")
    if multiselect_cols:
        multiselect_cols = list(dict.fromkeys(multiselect_cols))  # 去重并保持顺序
        print("\n   🧩 检测到岗位多选列:")
        for col in multiselect_cols[:10]:
            display_name = raw_header_map.get(col, col)
            print(f"      - {col} ({display_name})")
        if len(multiselect_cols) > 10:
            print(f"      ... 共 {len(multiselect_cols)} 列")

        job_indices, mapping = _load_multiselect_jobs(df, multiselect_cols, raw_header_map)
        return torch.from_numpy(job_indices), mapping

    # 否则回退到启发式单列
    column = detect_job_column(df)
    if column is None:
        raise ValueError("未找到岗位列，请通过 --job-column 指定")
    else:
        display_name = raw_header_map.get(column, column)
        print(f"\n   ⚠️ 未检测到岗位多选列，将使用单列 `{column}` ({display_name}) 作为岗位类别来源")

    series = df[column].fillna("未知岗位")
    if series.dtype.kind in {"i", "u"}:
        codes = series.astype(int).values
        unique_vals = np.unique(codes)
        mapping = {str(val): idx for idx, val in enumerate(sorted(unique_vals))}
        remapped = np.vectorize(lambda v: mapping[str(v)])(codes)
    else:
        
        categories = series.astype("category")
        remapped = categories.cat.codes.values
        mapping = {
            str(cat): int(code)
            for cat, code in zip(categories.cat.categories, range(len(categories.cat.categories)))
        }

    remapped = remapped.astype(np.int64)
    if remapped.min() < 0:
        raise ValueError("岗位编码存在缺失，无法转换，请检查原始CSV")

    return torch.from_numpy(remapped), mapping


def print_config(args, job_mapping: Dict[str, int], data: Data):
    """打印训练配置"""
    print("\n" + "=" * 70)
    print("📋 训练配置 v4")
    print("=" * 70)
    print("\n数据:")
    print(f"   data_path: {args.data_path}")
    print(f"   raw_csv: {args.raw_csv}")
    print(f"   节点: {data.x.size(0)}")
    print(f"   特征: {data.x.size(1)}")
    print(f"   边: {data.edge_index.size(1)}")
    print(f"   训练/验证/测试: {data.train_mask.sum().item()}/{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")

    print("\n岗位映射:")
    for name, idx in sorted(job_mapping.items(), key=lambda kv: kv[1]):
        print(f"   {idx:2d} ↔ {name}")

    print("\n模型:")
    print(f"   hidden_dim: {args.hidden_dim}")
    print(f"   num_layers: {args.num_layers}")
    print(f"   dropout: {args.dropout}")
    print(f"   edge_dropout: {args.edge_dropout}")
    print(f"   feature_dropout: {args.feature_dropout}")
    print(f"   job_embed_dim: {args.job_embed_dim}")
    print(f"   mlp_hidden_dim: {args.mlp_hidden_dim}")

    print("\n训练:")
    print(f"   lr: {args.lr}")
    print(f"   weight_decay: {args.weight_decay}")
    print(f"   epochs: {args.epochs}")
    print(f"   patience: {args.patience}")
    print(f"   device: {args.device}")
    print(f"   use_focal_loss: {args.use_focal_loss} (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    print("=" * 70)


@torch.no_grad()
def evaluate_split(
    model: EmployeeJobFusionModel,
    data: Data,
    job_indices: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    model.eval()
    mask = mask.bool()
    logits, _, _ = model(data.x, data.edge_index, job_indices)
    logits = logits[mask]
    labels = data.y[mask].float()
    loss = loss_fn(logits, labels).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    metrics = compute_metrics(labels.cpu().numpy(), probs, threshold=0.5)
    metrics.update({"loss": loss})
    return metrics


def train(
    model: EmployeeJobFusionModel,
    data: Data,
    job_indices: torch.Tensor,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    criterion: nn.Module,
    bce_loss: nn.Module,
    args,
    save_dir: Path,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_recall": [],
        "lr": [],
    }

    best_state = None
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience_counter = 0

    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    labels = data.y.float()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, _, _ = model(data.x, data.edge_index, job_indices)
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]

        loss = criterion(train_logits, train_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 验证
        val_metrics = evaluate_split(model, data, job_indices, val_mask, bce_loss)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_recall"].append(val_metrics["recall"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        improved = False
        if val_metrics["loss"] < best_val_loss - 1e-4:
            improved = True
        elif abs(val_metrics["loss"] - best_val_loss) <= 1e-4 and val_metrics["f1"] > best_val_f1:
            improved = True

        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1"]
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            patience_counter = 0
            torch.save(best_state, save_dir / "best_model.pt")
            print(
                f"   ⭐ 保存最佳模型 (epoch={epoch}, val_loss={best_val_loss:.4f}, val_f1={best_val_f1:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"   🛑 早停触发，epoch={epoch}")
                break

    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": args.epochs,
            "val_metrics": evaluate_split(model, data, job_indices, val_mask, bce_loss),
        }
        torch.save(best_state, save_dir / "best_model.pt")

    model.load_state_dict(best_state["model"])
    return history, best_state


def parse_args():
    parser = argparse.ArgumentParser(description="GCN v4 - 离职预测（岗位融合）")

    # 数据
    parser.add_argument("--data-path", type=str, default="data/processed/homo_graph.pt")
    parser.add_argument("--raw-csv", type=str, default="data/raw/originaldata.csv")
    parser.add_argument("--job-column", type=str, default=None, help="原始CSV中的岗位列名称")
    parser.add_argument("--encoding", type=str, default="gbk")
    parser.add_argument("--skiprows", type=int, default=1)

    # 模型
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--edge-dropout", type=float, default=0.0)
    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--job-embed-dim", type=int, default=32)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)

    # 训练
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="outputs/models")
    parser.add_argument("--eval-dir", type=str, default="outputs/evaluation")

    # Loss
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data: Data = torch.load(args.data_path)
    data = data.to(device)
    if data.y.dim() > 1:
        data.y = data.y.view(-1)

    job_indices, job_mapping = load_job_indices(
        Path(args.raw_csv),
        num_nodes=data.x.size(0),
        job_column=args.job_column,
        encoding=args.encoding,
        skiprows=args.skiprows,
    )
    job_indices = job_indices.to(device)
    num_jobs = int(job_indices.max().item() + 1)

    print_config(args, job_mapping, data)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    print(f"\n🆔 Run ID: {run_id}")
    save_dir = Path(args.save_dir) / run_id
    eval_dir = Path(args.eval_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    encoder = EmployeeGCNEncoder(
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        feature_dropout=args.feature_dropout,
    )
    model = EmployeeJobFusionModel(
        encoder=encoder,
        num_jobs=num_jobs,
        job_embed_dim=args.job_embed_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.dropout,
    ).to(device)

    num_pos = data.y[data.train_mask].sum().item()
    num_total = data.train_mask.sum().item()
    pos_weight_value = (num_total - num_pos) / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], device=device)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, pos_weight=pos_weight)
    else:
        criterion = bce_loss

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    history, best_state = train(
        model=model,
        data=data,
        job_indices=job_indices,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        bce_loss=bce_loss,
        args=args,
        save_dir=save_dir,
    )

    # 最佳模型评估
    val_metrics = evaluate_split(model, data, job_indices, data.val_mask, bce_loss)
    test_metrics = evaluate_split(model, data, job_indices, data.test_mask, bce_loss)

    print("\n" + "=" * 70)
    print("✅ 训练完成 (GCN v4)")
    print("=" * 70)
    print(f"最佳Epoch: {best_state['epoch']}")
    print(f"验证集 - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Recall: {val_metrics['recall']:.4f}")
    print(f"测试集 -  Loss: {test_metrics['loss']:.4f}, F1: {test_metrics['f1']:.4f}, Recall: {test_metrics['recall']:.4f}")
    print("=" * 70)

    # 保存历史
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    results = {
        "run_id": run_id,
        "config": vars(args),
        "best_epoch": best_state["epoch"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pos_weight": pos_weight_value,
        "num_jobs": num_jobs,
        "job_mapping": job_mapping,
    }
    with open(eval_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(eval_dir / "job_mapping.json", "w") as f:
        json.dump(job_mapping, f, ensure_ascii=False, indent=2)

    print(f"\n💾 最佳模型已保存至: {save_dir / 'best_model.pt'}")
    print(f"📊 评估结果已保存至: {eval_dir / 'results.json'}")


if __name__ == "__main__":
    main()
