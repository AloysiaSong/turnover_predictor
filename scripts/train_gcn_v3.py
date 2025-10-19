"""
多任务GCN训练脚本 v3 - 离职预测 + 岗位偏好排序
========================================================
Loss = α × 离职分类Loss + β × 岗位偏好Ranking Loss

岗位偏好Loss: Pairwise Ranking Loss
对每个员工的7个岗位进行两两比较:
  如果 rank(岗位i) < rank(岗位j)，则 score(员工,岗位i) > score(员工,岗位j)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    ndcg_score
)
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultiTaskGCN(nn.Module):
    """
    多任务GCN: 离职预测 + 岗位偏好排序
    """
    def __init__(
        self,
        in_features,
        hidden_dims=[128, 64, 32],
        n_positions=7,
        position_embed_dim=32,
        dropout=0.5,
        preference_loss_type='pairwise'
    ):
        super().__init__()
        
        # GCN编码器
        self.convs = nn.ModuleList()
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            self.convs.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.final_dim = hidden_dims[-1]
        
        # 岗位embedding
        self.position_embedding = nn.Embedding(n_positions, position_embed_dim)
        
        # 离职预测头
        self.turnover_head = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.final_dim // 2, 1)
        )
        
        # 偏好预测: 投影到相同空间
        self.employee_proj = nn.Linear(self.final_dim, position_embed_dim)
        
        self.dropout = dropout
        self.n_positions = n_positions
        self.preference_loss_type = preference_loss_type
        
    def encode_employees(self, x, edge_index):
        """编码员工特征"""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            if i < len(self.convs) - 1:  # 最后一层不dropout
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        Returns:
            employee_emb: (N, final_dim) 员工embedding
            turnover_logits: (N,) 离职预测logits
            preference_scores: (N, n_positions) 岗位偏好得分
        """
        # 编码员工
        employee_emb = self.encode_employees(x, edge_index)  # (N, final_dim)
        
        # 离职预测
        turnover_logits = self.turnover_head(employee_emb).squeeze()  # (N,)
        
        # 岗位偏好得分
        employee_proj = self.employee_proj(employee_emb)  # (N, position_embed_dim)
        position_emb = self.position_embedding.weight  # (n_positions, position_embed_dim)
        preference_scores = employee_proj @ position_emb.T  # (N, n_positions)
        
        return employee_emb, turnover_logits, preference_scores
    
    def compute_loss(
        self,
        turnover_logits,
        turnover_labels,
        preference_scores,
        preference_ranks,
        alpha=0.5,
        beta=0.5,
        pos_weight=None
    ):
        """
        计算多任务损失
        
        Args:
            turnover_logits: (N,) 离职预测logits
            turnover_labels: (N,) 离职标签 0/1
            preference_scores: (N, K) 岗位得分 K=n_positions
            preference_ranks: (N, K) 岗位排序 1-7 (1=最偏好)
            alpha: 离职loss权重
            beta: 偏好loss权重
            pos_weight: 正样本权重
        
        Returns:
            total_loss, turnover_loss, preference_loss
        """
        # Loss 1: 离职分类
        if pos_weight is not None:
            turnover_loss = F.binary_cross_entropy_with_logits(
                turnover_logits,
                turnover_labels.float(),
                pos_weight=pos_weight
            )
        else:
            turnover_loss = F.binary_cross_entropy_with_logits(
                turnover_logits,
                turnover_labels.float()
            )
        
        # Loss 2: 岗位偏好排序
        if self.preference_loss_type == 'pairwise':
            preference_loss = self._pairwise_ranking_loss(
                preference_scores, preference_ranks
            )
        elif self.preference_loss_type == 'listnet':
            preference_loss = self._listnet_loss(
                preference_scores, preference_ranks
            )
        else:
            raise ValueError(f"Unknown loss type: {self.preference_loss_type}")
        
        # 总损失
        total_loss = alpha * turnover_loss + beta * preference_loss
        
        return total_loss, turnover_loss, preference_loss
    
    def _pairwise_ranking_loss(self, scores, ranks):
        """
        Pairwise Ranking Loss (向量化实现)
        
        对于每个员工，如果 rank[i] < rank[j] (岗位i更偏好)
        则希望 score[i] > score[j]
        
        Loss = max(0, margin - (score[i] - score[j]))
        """
        N, K = scores.shape  # N=员工数, K=岗位数
        
        # 构建pairwise mask: (N, K, K)
        # mask[n, i, j] = 1 if rank[n, i] < rank[n, j]
        ranks_i = ranks.unsqueeze(2)  # (N, K, 1)
        ranks_j = ranks.unsqueeze(1)  # (N, 1, K)
        prefer_mask = (ranks_i < ranks_j).float()  # (N, K, K)
        
        # 计算所有pairwise得分差值: (N, K, K)
        scores_i = scores.unsqueeze(2)  # (N, K, 1)
        scores_j = scores.unsqueeze(1)  # (N, 1, K)
        score_diff = scores_i - scores_j  # (N, K, K)
        
        # Margin ranking loss: max(0, margin - score_diff)
        margin = 1.0
        margin_loss = F.relu(margin - score_diff) * prefer_mask
        
        # 平均
        num_pairs = prefer_mask.sum()
        if num_pairs > 0:
            loss = margin_loss.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, device=scores.device)
        
        return loss
    
    def _listnet_loss(self, scores, ranks):
        """
        ListNet Loss: 直接优化排序分布
        
        使用KL散度比较真实排序分布和预测排序分布
        """
        # 真实排序 → 概率分布 (rank越小，概率越大)
        # 使用 softmax(-rank) 来转换
        true_probs = F.softmax(-ranks.float(), dim=1)  # (N, K)
        
        # 预测得分 → 概率分布
        pred_probs = F.softmax(scores, dim=1)  # (N, K)
        
        # KL散度: KL(true || pred)
        loss = F.kl_div(
            pred_probs.log(),
            true_probs,
            reduction='batchmean'
        )
        
        return loss


def set_seed(seed: int):
    """固定随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_data(data_path: str):
    """
    加载数据
    
    需要包含:
        - x: 员工特征
        - y: 离职标签
        - edge_index: 图边
        - preference_ranks: 岗位偏好排序 (N, 7)
        - train_mask, val_mask, test_mask
    """
    print("\n1. 加载数据...")
    data = torch.load(data_path)
    
    print(f"   ✓ 节点: {data.x.shape[0]}")
    print(f"   ✓ 边: {data.edge_index.shape[1]}")
    print(f"   ✓ 特征: {data.x.shape[1]}")
    print(f"   ✓ 训练集: {data.train_mask.sum()}")
    print(f"   ✓ 验证集: {data.val_mask.sum()}")
    print(f"   ✓ 测试集: {data.test_mask.sum()}")
    
    # 检查是否有preference_ranks
    if not hasattr(data, 'preference_ranks'):
        print("\n   ⚠️  数据中没有preference_ranks，需要先准备岗位偏好数据！")
        print("   提示: 从原始数据中提取7个情景任务的岗位偏好排序")
        raise ValueError("Missing preference_ranks in data")
    
    print(f"   ✓ 岗位数: {data.preference_ranks.shape[1]}")
    
    return data


def evaluate_turnover(model, data, mask, device):
    """评估离职预测性能"""
    model.eval()
    with torch.no_grad():
        _, turnover_logits, _ = model(data.x.to(device), data.edge_index.to(device))
        
        logits = turnover_logits[mask].cpu()
        labels = data.y[mask].cpu()
        
        probs = torch.sigmoid(logits).numpy()
        labels_np = labels.numpy()
        
        aupr = average_precision_score(labels_np, probs)
        auroc = roc_auc_score(labels_np, probs)
        
        return aupr, auroc, probs, labels_np


def evaluate_preference(model, data, mask, device):
    """
    评估岗位偏好预测性能
    
    使用NDCG@K和Kendall's Tau
    """
    model.eval()
    with torch.no_grad():
        _, _, preference_scores = model(data.x.to(device), data.edge_index.to(device))
        
        scores = preference_scores[mask].cpu().numpy()  # (N, K)
        ranks = data.preference_ranks[mask].cpu().numpy()  # (N, K)
        
        # NDCG@3: 评估Top-3推荐质量
        # 需要将rank转为relevance score (rank 1 → score 7)
        relevance = 8 - ranks  # rank 1-7 → relevance 7-1
        
        ndcg3_scores = []
        ndcg5_scores = []
        
        for i in range(len(scores)):
            # NDCG@3
            ndcg3 = ndcg_score([relevance[i]], [scores[i]], k=3)
            ndcg3_scores.append(ndcg3)
            
            # NDCG@5
            ndcg5 = ndcg_score([relevance[i]], [scores[i]], k=5)
            ndcg5_scores.append(ndcg5)
        
        ndcg3 = np.mean(ndcg3_scores)
        ndcg5 = np.mean(ndcg5_scores)
        
        # Pairwise accuracy
        pairwise_acc = compute_pairwise_accuracy(scores, ranks)
        
        return ndcg3, ndcg5, pairwise_acc


def compute_pairwise_accuracy(scores, ranks):
    """
    计算成对比较的准确率
    
    对于每对岗位(i,j)，如果rank[i] < rank[j]，检查是否score[i] > score[j]
    """
    N, K = scores.shape
    correct = 0
    total = 0
    
    for n in range(N):
        for i in range(K):
            for j in range(i+1, K):
                if ranks[n, i] != ranks[n, j]:  # 排名不同
                    total += 1
                    # 检查预测是否正确
                    if ranks[n, i] < ranks[n, j]:  # i更偏好
                        if scores[n, i] > scores[n, j]:
                            correct += 1
                    else:  # j更偏好
                        if scores[n, j] > scores[n, i]:
                            correct += 1
    
    return correct / total if total > 0 else 0.0


def find_best_threshold(y_true, y_pred_proba):
    """在验证集上寻找最优阈值"""
    best_threshold = 0.5
    best_f1 = 0.0
    
    thresholds = np.arange(0.1, 0.9, 0.02)
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    return best_threshold, best_f1


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """计算所有评估指标"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'aupr': average_precision_score(y_true, y_pred_proba),
        'auroc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    return metrics


def train_epoch(model, data, optimizer, criterion_fn, device, alpha, beta, pos_weight):
    """训练一个epoch"""
    model.train()
    
    # 前向传播
    _, turnover_logits, preference_scores = model(
        data.x.to(device), 
        data.edge_index.to(device)
    )
    
    # 只在训练集上计算loss
    train_mask = data.train_mask
    
    total_loss, turnover_loss, preference_loss = model.compute_loss(
        turnover_logits[train_mask],
        data.y[train_mask].to(device),
        preference_scores[train_mask],
        data.preference_ranks[train_mask].to(device),
        alpha=alpha,
        beta=beta,
        pos_weight=pos_weight
    )
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return (
        total_loss.item(),
        turnover_loss.item(),
        preference_loss.item()
    )


def main():
    parser = argparse.ArgumentParser(description='多任务GCN训练 - 离职+岗位偏好')
    
    # 数据参数
    parser.add_argument('--data-path', type=str,
                       default='data/processed/homo_graph_with_preferences.pt',
                       help='数据路径（需包含preference_ranks）')
    
    # 模型参数
    parser.add_argument('--architecture', type=str, default='default',
                       choices=['shallow', 'default', 'deep'],
                       help='模型架构')
    parser.add_argument('--position-embed-dim', type=int, default=32,
                       help='岗位embedding维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    parser.add_argument('--preference-loss', type=str, default='pairwise',
                       choices=['pairwise', 'listnet'],
                       help='偏好loss类型')
    
    # 损失权重
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='离职loss权重')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='偏好loss权重')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.01,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='权重衰减')
    parser.add_argument('--epochs', type=int, default=200,
                       help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停patience')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save-dir', type=str,
                       default='outputs/multitask_gcn',
                       help='保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print("\n" + "="*70)
    print("🎯 多任务GCN训练配置")
    print("="*70)
    print(f"\n数据: {args.data_path}")
    print(f"\n模型:")
    print(f"   architecture: {args.architecture}")
    print(f"   position_embed_dim: {args.position_embed_dim}")
    print(f"   dropout: {args.dropout}")
    print(f"   preference_loss: {args.preference_loss}")
    print(f"\n损失权重:")
    print(f"   α (离职): {args.alpha}")
    print(f"   β (偏好): {args.beta}")
    print(f"\n训练:")
    print(f"   lr: {args.lr}")
    print(f"   weight_decay: {args.weight_decay}")
    print(f"   epochs: {args.epochs}")
    print(f"   patience: {args.patience}")
    print(f"\n其他:")
    print(f"   device: {args.device}")
    print(f"   seed: {args.seed}")
    print("="*70)
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"run_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    data = load_data(args.data_path)
    
    # 创建模型
    print("\n2. 创建多任务GCN模型...")
    architectures = {
        'shallow': [64, 32],
        'default': [128, 64, 32],
        'deep': [256, 128, 64, 32]
    }
    hidden_dims = architectures[args.architecture]
    
    model = MultiTaskGCN(
        in_features=data.x.shape[1],
        hidden_dims=hidden_dims,
        n_positions=data.preference_ranks.shape[1],
        position_embed_dim=args.position_embed_dim,
        dropout=args.dropout,
        preference_loss_type=args.preference_loss
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 架构: {hidden_dims}")
    print(f"   ✓ 参数量: {num_params:,}")
    print(f"   ✓ 岗位数: {data.preference_ranks.shape[1]}")
    print(f"   ✓ 岗位embedding维度: {args.position_embed_dim}")
    
    device = torch.device(args.device)
    model = model.to(device)
    
    # 计算pos_weight
    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    
    print(f"\n3. 训练配置...")
    print(f"   ✓ 正样本: {num_pos} ({num_pos/(num_pos+num_neg)*100:.1f}%)")
    print(f"   ✓ 负样本: {num_neg} ({num_neg/(num_pos+num_neg)*100:.1f}%)")
    print(f"   ✓ pos_weight: {pos_weight.item():.2f}")
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # 训练
    print("\n4. 开始训练...")
    print("="*70)
    
    best_val_aupr = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_total_loss': [],
        'train_turnover_loss': [],
        'train_preference_loss': [],
        'val_aupr': [],
        'val_auroc': [],
        'val_ndcg3': [],
        'val_ndcg5': [],
        'lr': []
    }
    
    for epoch in tqdm(range(args.epochs), desc='Training'):
        # 训练
        train_total_loss, train_turn_loss, train_pref_loss = train_epoch(
            model, data, optimizer, None, device,
            args.alpha, args.beta, pos_weight
        )
        
        # 验证
        val_aupr, val_auroc, _, _ = evaluate_turnover(
            model, data, data.val_mask, device
        )
        val_ndcg3, val_ndcg5, val_pairwise = evaluate_preference(
            model, data, data.val_mask, device
        )
        
        # 学习率调整
        scheduler.step(val_aupr)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_total_loss'].append(train_total_loss)
        history['train_turnover_loss'].append(train_turn_loss)
        history['train_preference_loss'].append(train_pref_loss)
        history['val_aupr'].append(val_aupr)
        history['val_auroc'].append(val_auroc)
        history['val_ndcg3'].append(val_ndcg3)
        history['val_ndcg5'].append(val_ndcg5)
        history['lr'].append(current_lr)
        
        # 打印进度（每10轮）
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1:3d}:")
            print(f"  Train - Total: {train_total_loss:.4f}, "
                  f"Turnover: {train_turn_loss:.4f}, "
                  f"Preference: {train_pref_loss:.4f}")
            print(f"  Val   - AUPR: {val_aupr:.4f}, AUROC: {val_auroc:.4f}")
            print(f"          NDCG@3: {val_ndcg3:.4f}, NDCG@5: {val_ndcg5:.4f}")
            print(f"          LR: {current_lr:.6f}")
        
        # 早停检查
        if val_aupr > best_val_aupr:
            best_val_aupr = val_aupr
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            if (epoch + 1) % 10 == 0:
                print(f"  ⭐ 新的最佳模型!")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n   🛑 早停触发于epoch {epoch+1}")
            print(f"   📌 最佳epoch: {best_epoch+1}")
            break
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    print(f"   最佳Epoch: {best_epoch+1}")
    print(f"   最佳Val AUPR: {best_val_aupr:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    
    # 在验证集上找最优阈值
    print("\n5. 在验证集上寻找最优阈值...")
    _, _, val_probs, val_labels = evaluate_turnover(
        model, data, data.val_mask, device
    )
    best_threshold, best_f1 = find_best_threshold(val_labels, val_probs)
    print(f"   ✓ 最优阈值: {best_threshold:.2f} (Val F1={best_f1:.4f})")
    
    # 在测试集上评估
    print("\n6. 在测试集上评估...")
    print("\n📊 离职预测性能:")
    _, _, test_probs, test_labels = evaluate_turnover(
        model, data, data.test_mask, device
    )
    test_turn_metrics = compute_metrics(test_labels, test_probs, best_threshold)
    
    print(f"   AUPR:      {test_turn_metrics['aupr']:.4f}")
    print(f"   AUROC:     {test_turn_metrics['auroc']:.4f}")
    print(f"   F1:        {test_turn_metrics['f1']:.4f}")
    print(f"   Precision: {test_turn_metrics['precision']:.4f}")
    print(f"   Recall:    {test_turn_metrics['recall']:.4f}")
    
    print("\n📊 岗位偏好性能:")
    test_ndcg3, test_ndcg5, test_pairwise = evaluate_preference(
        model, data, data.test_mask, device
    )
    print(f"   NDCG@3:         {test_ndcg3:.4f}")
    print(f"   NDCG@5:         {test_ndcg5:.4f}")
    print(f"   Pairwise Acc:   {test_pairwise:.4f}")
    
    # 保存结果
    print("\n7. 保存结果...")
    results = {
        'config': vars(args),
        'model_params': num_params,
        'best_epoch': best_epoch + 1,
        'best_val_aupr': best_val_aupr,
        'best_threshold': best_threshold,
        'test_turnover_metrics': {k: float(v) for k, v in test_turn_metrics.items()},
        'test_preference_metrics': {
            'ndcg@3': float(test_ndcg3),
            'ndcg@5': float(test_ndcg5),
            'pairwise_accuracy': float(test_pairwise)
        },
        'training_history': history
    }
    
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ 结果已保存: {results_path}")
    
    # 生成训练曲线
    print("\n8. 生成训练曲线...")
    plot_training_curves(history, save_dir)
    
    print("\n" + "="*70)
    print("🎉 多任务GCN训练完成!")
    print("="*70)
    print(f"\n📊 最终性能总结:")
    print(f"\n离职预测:")
    print(f"   AUPR:  {test_turn_metrics['aupr']:.4f}")
    print(f"   F1:    {test_turn_metrics['f1']:.4f}")
    print(f"\n岗位偏好:")
    print(f"   NDCG@3: {test_ndcg3:.4f}")
    print(f"   NDCG@5: {test_ndcg5:.4f}")
    
    # 与单任务对比
    print(f"\n💡 与单任务GCN对比:")
    print(f"   单任务GCN: AUPR = 0.3153")
    print(f"   多任务GCN: AUPR = {test_turn_metrics['aupr']:.4f}")
    
    if test_turn_metrics['aupr'] > 0.3153:
        improvement = (test_turn_metrics['aupr'] - 0.3153) / 0.3153 * 100
        print(f"   ✅ 提升: +{improvement:.1f}%")
    else:
        decline = (0.3153 - test_turn_metrics['aupr']) / 0.3153 * 100
        print(f"   ⚠️  下降: -{decline:.1f}%")
    
    print("="*70 + "\n")


def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # 总损失
    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 离职损失
    axes[0, 1].plot(epochs, history['train_turnover_loss'], 'r-', label='Turnover Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Turnover Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 偏好损失
    axes[0, 2].plot(epochs, history['train_preference_loss'], 'g-', label='Preference Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Training Preference Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Val AUPR
    axes[1, 0].plot(epochs, history['val_aupr'], 'b-', label='Val AUPR')
    axes[1, 0].axhline(y=0.3153, color='r', linestyle='--', label='单任务GCN')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUPR')
    axes[1, 0].set_title('Validation AUPR')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Val NDCG
    axes[1, 1].plot(epochs, history['val_ndcg3'], 'r-', label='NDCG@3')
    axes[1, 1].plot(epochs, history['val_ndcg5'], 'g-', label='NDCG@5')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('NDCG')
    axes[1, 1].set_title('Validation NDCG')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Learning Rate
    axes[1, 2].plot(epochs, history['lr'], 'k-', label='Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('LR')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ 训练曲线已保存: {save_dir / 'training_curves.png'}")


if __name__ == '__main__':
    main()
