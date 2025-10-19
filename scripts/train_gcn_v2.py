"""
GCN训练脚本（完整版）
====================
功能：
1. EdgeDropout + FeatureDropout
2. 特征扩展（分箱+交互项）
3. Val_AUCPR早停
4. 阈值优化（仅在Val上）
5. Ablation Study
6. 完整CLI参数
"""

import torch
import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import random

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gcn import create_gcn_model
from src.models.trainer2 import GCNTrainer
from src.data.feature_pipeline import NodeFeatureTransformer


def set_seed(seed: int):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GCN训练 - 完整配置')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default='data/processed/homo_graph.pt',
                       help='数据路径')
    
    # 模型参数
    parser.add_argument('--architecture', type=str, default='default',
                       choices=['shallow', 'default', 'deep', 'very_deep'],
                       help='模型架构')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='标准dropout率')
    parser.add_argument('--edge-dropout', type=float, default=0.0,
                       help='边dropout率 (0.0-0.5)')
    parser.add_argument('--feature-dropout', type=float, default=0.0,
                       help='特征dropout率 (0.0-0.5)')
    
    # 特征扩展参数
    parser.add_argument('--feat-augment', type=str, default='off',
                       choices=['on', 'off'],
                       help='是否启用特征扩展')
    parser.add_argument('--topn', type=int, default=3,
                       help='Top-N重要特征数量')
    parser.add_argument('--bins', type=int, default=5,
                       help='分箱数量')
    parser.add_argument('--add-interactions', type=str, default='on',
                       choices=['on', 'off'],
                       help='是否添加交互项')
    parser.add_argument('--importance-metric', type=str, default='aupr',
                       choices=['aupr', 'mi', 'lr_coef'],
                       help='特征重要性度量')
    
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
    parser.add_argument('--save-dir', type=str, default='outputs/models',
                       help='模型保存目录')
    parser.add_argument('--eval-dir', type=str, default='outputs/evaluation',
                       help='评估结果保存目录')
    
    # Ablation Study
    parser.add_argument('--run-ablation', action='store_true',
                       help='运行ablation study')
    
    return parser.parse_args()


def print_config(args):
    """打印配置"""
    print("\n" + "="*70)
    print("📋 训练配置")
    print("="*70)
    
    print("\n数据:")
    print(f"   data_path: {args.data_path}")
    
    print("\n模型:")
    print(f"   architecture: {args.architecture}")
    print(f"   dropout: {args.dropout}")
    print(f"   edge_dropout: {args.edge_dropout}")
    print(f"   feature_dropout: {args.feature_dropout}")
    
    print("\n特征扩展:")
    print(f"   feat_augment: {args.feat_augment}")
    if args.feat_augment == 'on':
        print(f"   topn: {args.topn}")
        print(f"   bins: {args.bins}")
        print(f"   add_interactions: {args.add_interactions}")
        print(f"   importance_metric: {args.importance_metric}")
    
    print("\n训练:")
    print(f"   lr: {args.lr}")
    print(f"   weight_decay: {args.weight_decay}")
    print(f"   epochs: {args.epochs}")
    print(f"   patience: {args.patience}")
    
    print("\n其他:")
    print(f"   device: {args.device}")
    print(f"   seed: {args.seed}")
    
    print("\n" + "="*70)


def train_single_config(args, config_name: str):
    """训练单个配置"""
    
    print(f"\n{'='*70}")
    print(f"🚀 训练配置: {config_name}")
    print(f"{'='*70}")
    
    # 固定随机种子
    set_seed(args.seed)
    
    # 加载数据
    print("\n1. 加载数据...")
    data = torch.load(args.data_path)
    print(f"   ✓ 节点: {data.num_nodes}")
    print(f"   ✓ 边: {data.num_edges}")
    print(f"   ✓ 特征: {data.num_node_features}")
    print(f"   ✓ 训练集: {data.train_mask.sum()}")
    print(f"   ✓ 验证集: {data.val_mask.sum()}")
    print(f"   ✓ 测试集: {data.test_mask.sum()}")
    
    # 特征扩展
    feature_transformer = None
    if args.feat_augment == 'on':
        print("\n2. 特征扩展...")
        feature_transformer = NodeFeatureTransformer(
            topn=args.topn,
            n_bins=args.bins,
            add_interactions=(args.add_interactions == 'on'),
            importance_metric=args.importance_metric,
            random_state=args.seed
        )
        
        # 拟合并转换
        data.x = feature_transformer.fit_transform(data.x, data.y, data.train_mask)
        
        # 保存转换器
        transformer_path = Path(args.save_dir) / config_name / 'feature_transformer.pkl'
        transformer_path.parent.mkdir(parents=True, exist_ok=True)
        feature_transformer.save(str(transformer_path))
    
    # 创建模型
    print("\n3. 创建模型...")
    model = create_gcn_model(
        in_channels=data.num_node_features,
        architecture=args.architecture,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        feature_dropout=args.feature_dropout
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 参数量: {num_params:,}")
    print(f"   ✓ Edge Dropout: {args.edge_dropout}")
    print(f"   ✓ Feature Dropout: {args.feature_dropout}")
    
    # 创建训练器
    print("\n4. 创建训练器...")
    
    # 计算pos_weight
    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    pos_weight = num_neg / num_pos
    
    print(f"   ✓ 正样本: {num_pos} ({num_pos/(num_pos+num_neg)*100:.1f}%)")
    print(f"   ✓ 负样本: {num_neg} ({num_neg/(num_pos+num_neg)*100:.1f}%)")
    print(f"   ✓ pos_weight: {pos_weight:.2f}")
    
    trainer = GCNTrainer(
        model=model,
        data=data,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight
    )
    
    # 训练
    print("\n5. 开始训练...")
    save_dir = Path(args.save_dir) / config_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_dir=str(save_dir),
        verbose=True
        # 注：trainer已默认监控val_aupr
    )
    
    # 保存训练历史
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 评估
    print("\n6. 评估...")
    eval_dir = Path(args.eval_dir) / config_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # 在Val上找最优阈值
    print("\n   在验证集上扫描阈值...")
    val_probs, _, val_labels = trainer.predict(data.val_mask)
    
    best_threshold = 0.5
    best_f1 = 0.0
    threshold_results = []
    
    for t in np.arange(0.05, 0.75, 0.02):
        val_preds = (val_probs > t).astype(int)
        f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        threshold_results.append({
            'threshold': float(t),
            'f1': float(f1)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"   ✓ 最优阈值: {best_threshold:.2f} (Val F1={best_f1:.4f})")
    
    # 在Test上使用固定阈值
    print(f"\n   在测试集上评估 (阈值={best_threshold:.2f})...")
    test_probs, _, test_labels = trainer.predict(data.test_mask)
    
    from src.models.trainer import compute_metrics
    test_metrics = compute_metrics(test_labels, test_probs, threshold=best_threshold)
    
    print(f"   ✓ Test AUPR: {test_metrics['aupr']:.4f}")
    print(f"   ✓ Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"   ✓ Test F1: {test_metrics['f1']:.4f}")
    print(f"   ✓ Test Recall: {test_metrics['recall']:.4f}")
    print(f"   ✓ Test Precision: {test_metrics['precision']:.4f}")
    
    # 保存结果
    results = {
        'config_name': config_name,
        'config': vars(args),
        'best_threshold': float(best_threshold),
        'best_val_f1': float(best_f1),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'threshold_scan': threshold_results,
        'training_history': history
    }
    
    results_path = eval_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n   ✓ 结果已保存: {results_path}")
    
    return results


def run_ablation_study(args):
    """运行Ablation Study"""
    
    print("\n" + "="*70)
    print("🧪 Ablation Study")
    print("="*70)
    
    # 定义配置
    configs = [
        {
            'name': 'baseline',
            'edge_dropout': 0.0,
            'feature_dropout': 0.0,
            'feat_augment': 'off'
        },
        {
            'name': 'edge_dropout',
            'edge_dropout': 0.2,
            'feature_dropout': 0.0,
            'feat_augment': 'off'
        },
        {
            'name': 'feature_dropout',
            'edge_dropout': 0.0,
            'feature_dropout': 0.2,
            'feat_augment': 'off'
        },
        {
            'name': 'feat_augment',
            'edge_dropout': 0.0,
            'feature_dropout': 0.0,
            'feat_augment': 'on'
        },
        {
            'name': 'all_combined',
            'edge_dropout': 0.2,
            'feature_dropout': 0.2,
            'feat_augment': 'on'
        }
    ]
    
    all_results = []
    
    for config in configs:
        # 更新args
        args_copy = argparse.Namespace(**vars(args))
        args_copy.edge_dropout = config['edge_dropout']
        args_copy.feature_dropout = config['feature_dropout']
        args_copy.feat_augment = config['feat_augment']
        
        # 训练
        try:
            result = train_single_config(args_copy, config['name'])
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ 配置 {config['name']} 训练失败: {e}")
            continue
    
    # 生成对比表
    print("\n" + "="*70)
    print("📊 Ablation Study 对比")
    print("="*70)
    
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Config': result['config_name'],
            'Val_AUPR': result['training_history']['val_aupr'][-1] if 'val_aupr' in result['training_history'] else 0.0,
            'Test_AUPR': result['test_metrics']['aupr'],
            'Test_F1': result['test_metrics']['f1'],
            'Test_Recall': result['test_metrics']['recall'],
            'Test_Precision': result['test_metrics']['precision'],
            'Threshold': result['best_threshold']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 打印表格
    print("\n" + df.to_string(index=False))
    
    # 保存CSV
    csv_path = Path(args.eval_dir) / 'ablation_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 对比表已保存: {csv_path}")
    
    # 保存完整结果
    full_results_path = Path(args.eval_dir) / 'ablation_full_results.json'
    with open(full_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ 完整结果已保存: {full_results_path}")
    
    print("\n" + "="*70)
    print("✅ Ablation Study 完成！")
    print("="*70)


def main():
    """主函数"""
    args = parse_args()
    
    # 打印配置
    print_config(args)
    
    # 创建输出目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    
    if args.run_ablation:
        # 运行Ablation Study
        run_ablation_study(args)
    else:
        # 单次训练
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"run_{timestamp}"
        train_single_config(args, config_name)


if __name__ == '__main__':
    # 添加sklearn导入
    from sklearn.metrics import f1_score
    main()