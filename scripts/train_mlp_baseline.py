"""
MLP Baseline 训练脚本（无图版本）
===================================
目的: 作为GCN的关键对比 - 评估图结构是否有用
特点:
1. 与GCN使用完全相同的特征（47维）
2. 相同的训练/验证/测试划分
3. 相同的评估指标和阈值优化
4. 唯一区别：不使用图结构
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MLP(nn.Module):
    """简单的MLP模型"""
    def __init__(self, in_features, hidden_dims, dropout=0.5):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


def set_seed(seed: int):
    """固定随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_data(data_path: str):
    """加载图数据并提取特征"""
    print("\n1. 加载数据...")
    data = torch.load(data_path)
    
    X = data.x  # (500, 47)
    y = data.y  # (500,)
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    print(f"   ✓ 特征维度: {X.shape}")
    print(f"   ✓ 训练集: {train_mask.sum()}")
    print(f"   ✓ 验证集: {val_mask.sum()}")
    print(f"   ✓ 测试集: {test_mask.sum()}")
    print(f"   ✓ 离职率: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
    
    return X, y, train_mask, val_mask, test_mask


def create_dataloaders(X, y, train_mask, val_mask, test_mask, batch_size=32):
    """创建数据加载器"""
    train_dataset = TensorDataset(X[train_mask], y[train_mask])
    val_dataset = TensorDataset(X[val_mask], y[val_mask])
    test_dataset = TensorDataset(X[test_mask], y[test_mask])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            probs = torch.sigmoid(output).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    aupr = average_precision_score(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)
    
    return aupr, auroc, all_probs, all_labels


def find_best_threshold(y_true, y_pred_proba):
    """在验证集上寻找最优阈值"""
    best_threshold = 0.5
    best_f1 = 0.0
    
    thresholds = np.arange(0.1, 0.9, 0.02)
    threshold_results = []
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        threshold_results.append({
            'threshold': float(t),
            'f1': float(f1)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    return best_threshold, best_f1, threshold_results


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


def plot_curves(y_true, y_pred_proba, save_dir):
    """绘制PR和ROC曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    aupr = average_precision_score(y_true, y_pred_proba)
    
    axes[0].plot(recall, precision, 'b-', linewidth=2, label=f'MLP (AUPR={aupr:.3f})')
    axes[0].plot([0, 1], [y_true.sum()/len(y_true)]*2, 'r--', label='Random')
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'MLP (AUROC={auroc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'r--', label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ 曲线已保存: {save_dir / 'curves.png'}")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Leave', 'Leave'],
                yticklabels=['Not Leave', 'Leave'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - MLP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ 混淆矩阵已保存: {save_dir / 'confusion_matrix.png'}")


def main():
    parser = argparse.ArgumentParser(description='MLP Baseline训练')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, 
                       default='data/processed/homo_graph.pt',
                       help='数据路径')
    
    # 模型参数
    parser.add_argument('--architecture', type=str, default='default',
                       choices=['shallow', 'default', 'deep'],
                       help='模型架构')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.01,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='权重衰减')
    parser.add_argument('--epochs', type=int, default=200,
                       help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停patience')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save-dir', type=str, 
                       default='outputs/mlp_baseline',
                       help='保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print("\n" + "="*70)
    print("📋 MLP Baseline 训练配置（无图版本）")
    print("="*70)
    print(f"\n数据: {args.data_path}")
    print(f"模型:")
    print(f"   architecture: {args.architecture}")
    print(f"   dropout: {args.dropout}")
    print(f"训练:")
    print(f"   lr: {args.lr}")
    print(f"   weight_decay: {args.weight_decay}")
    print(f"   epochs: {args.epochs}")
    print(f"   patience: {args.patience}")
    print(f"   batch_size: {args.batch_size}")
    print(f"其他:")
    print(f"   device: {args.device}")
    print(f"   seed: {args.seed}")
    print("="*70)
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    X, y, train_mask, val_mask, test_mask = load_data(args.data_path)
    
    # 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y, train_mask, val_mask, test_mask, args.batch_size
    )
    
    # 创建模型
    print("\n3. 创建MLP模型...")
    architectures = {
        'shallow': [64, 32],
        'default': [128, 64, 32],
        'deep': [256, 128, 64, 32]
    }
    hidden_dims = architectures[args.architecture]
    
    model = MLP(X.shape[1], hidden_dims, args.dropout)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 架构: {hidden_dims}")
    print(f"   ✓ 参数量: {num_params:,}")
    print(f"   ✓ Dropout: {args.dropout}")
    
    device = torch.device(args.device)
    model = model.to(device)
    
    # 计算pos_weight
    num_pos = y[train_mask].sum().item()
    num_neg = train_mask.sum().item() - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    
    print(f"\n4. 训练配置...")
    print(f"   ✓ 正样本: {num_pos} ({num_pos/(num_pos+num_neg)*100:.1f}%)")
    print(f"   ✓ 负样本: {num_neg} ({num_neg/(num_pos+num_neg)*100:.1f}%)")
    print(f"   ✓ pos_weight: {pos_weight.item():.2f}")
    
    # 定义损失和优化器
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # 训练
    print("\n5. 开始训练...")
    best_val_aupr = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_aupr': [], 'val_auroc': []}
    
    for epoch in tqdm(range(args.epochs), desc='Training'):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_aupr, val_auroc, _, _ = evaluate(model, val_loader, device)
        
        # 学习率调整
        scheduler.step(val_aupr)
        
        # 记录历史
        history['train_loss'].append(float(train_loss))
        history['val_aupr'].append(float(val_aupr))
        history['val_auroc'].append(float(val_auroc))
        
        # 早停检查
        if val_aupr > best_val_aupr:
            best_val_aupr = val_aupr
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n   ✓ 早停触发于epoch {epoch+1}")
            print(f"   ✓ 最佳epoch: {best_epoch+1}")
            break
    
    print(f"\n   ✓ 训练完成!")
    print(f"   ✓ 最佳Val AUPR: {best_val_aupr:.4f} (epoch {best_epoch+1})")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    
    # 在验证集上找最优阈值
    print("\n6. 在验证集上寻找最优阈值...")
    _, _, val_probs, val_labels = evaluate(model, val_loader, device)
    best_threshold, best_f1, threshold_results = find_best_threshold(val_labels, val_probs)
    print(f"   ✓ 最优阈值: {best_threshold:.2f} (Val F1={best_f1:.4f})")
    
    # 在测试集上评估
    print("\n7. 在测试集上评估...")
    _, _, test_probs, test_labels = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs, best_threshold)
    
    print(f"\n   TEST集性能 (阈值={best_threshold:.2f}):")
    print(f"      AUPR:      {test_metrics['aupr']:.4f}")
    print(f"      AUROC:     {test_metrics['auroc']:.4f}")
    print(f"      F1:        {test_metrics['f1']:.4f}")
    print(f"      Precision: {test_metrics['precision']:.4f}")
    print(f"      Recall:    {test_metrics['recall']:.4f}")
    
    # 生成可视化
    print("\n8. 生成可视化...")
    plot_curves(test_labels, test_probs, save_dir)
    
    test_pred = (test_probs >= best_threshold).astype(int)
    plot_confusion_matrix(test_labels, test_pred, save_dir)
    
    # 保存结果
    print("\n9. 保存结果...")
    results = {
        'model': 'MLP',
        'architecture': args.architecture,
        'hidden_dims': hidden_dims,
        'seed': args.seed,
        'num_params': num_params,
        'best_epoch': best_epoch + 1,
        'best_val_aupr': float(best_val_aupr),
        'best_threshold': float(best_threshold),
        'best_val_f1': float(best_f1),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'threshold_scan': threshold_results,
        'training_history': history
    }
    
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ 结果已保存: {results_path}")
    
    # 生成文本报告
    report_path = save_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MLP Baseline - 评估报告（无图版本）\n")
        f.write("="*70 + "\n\n")
        f.write(f"1. 模型配置\n")
        f.write(f"   - 架构: {args.architecture} {hidden_dims}\n")
        f.write(f"   - 参数量: {num_params:,}\n")
        f.write(f"   - Dropout: {args.dropout}\n")
        f.write(f"   - 最佳Epoch: {best_epoch+1}\n\n")
        f.write(f"2. 验证集性能\n")
        f.write(f"   - 最佳AUPR: {best_val_aupr:.4f}\n")
        f.write(f"   - 最优阈值: {best_threshold:.2f}\n")
        f.write(f"   - 最优F1: {best_f1:.4f}\n\n")
        f.write(f"3. 测试集性能\n")
        f.write(f"   - AUPR:      {test_metrics['aupr']:.4f}\n")
        f.write(f"   - AUROC:     {test_metrics['auroc']:.4f}\n")
        f.write(f"   - F1 Score:  {test_metrics['f1']:.4f}\n")
        f.write(f"   - Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"   - Recall:    {test_metrics['recall']:.4f}\n\n")
        f.write("="*70 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"   ✓ 评估报告已保存: {report_path}")
    
    print("\n" + "="*70)
    print("✅ MLP Baseline 训练完成!")
    print("="*70)
    print(f"\n📊 测试集性能总结:")
    print(f"   AUPR:      {test_metrics['aupr']:.4f}")
    print(f"   AUROC:     {test_metrics['auroc']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    
    # 与GCN和XGBoost对比
    print("\n📈 与其他模型对比:")
    print("   GCN (w/ graph):")
    print("      AUPR:      0.3153")
    print("      AUROC:     0.8345")
    print("      F1:        0.4444")
    print(f"\n   MLP (w/o graph):")
    print(f"      AUPR:      {test_metrics['aupr']:.4f}")
    print(f"      AUROC:     {test_metrics['auroc']:.4f}")
    print(f"      F1:        {test_metrics['f1']:.4f}")
    print("\n   XGBoost:")
    print("      AUPR:      0.6805")
    print("      AUROC:     0.8723")
    print("      F1:        0.5926")
    
    print("\n💡 关键洞察:")
    if test_metrics['aupr'] > 0.315:
        print("   ✓ MLP > GCN: 图结构引入噪声，反而降低性能！")
    elif test_metrics['aupr'] < 0.315:
        print("   ✓ GCN > MLP: 图结构确实有帮助，但需要进一步优化")
    else:
        print("   ✓ MLP ≈ GCN: 图结构作用不明显")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
