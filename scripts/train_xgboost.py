"""
XGBoost Baseline 训练脚本
========================
目标: 作为GCN的强baseline对比
特点:
1. 使用相同的47维特征
2. 网格搜索最优超参数
3. 使用AUPR作为评估指标
4. 早停机制
5. 与GCN相同的数据划分
"""

import numpy as np
import pandas as pd
import xgboost as xgb
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
import torch
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed: int):
    """固定随机种子"""
    np.random.seed(seed)


def load_data(data_path: str):
    """
    加载处理好的图数据
    返回特征矩阵、标签和mask
    """
    print("\n1. 加载数据...")
    data = torch.load(data_path)
    
    # 提取numpy数组
    X = data.x.numpy()  # (500, 47)
    y = data.y.numpy()  # (500,)
    
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    print(f"   ✓ 特征维度: {X.shape}")
    print(f"   ✓ 训练集: {train_mask.sum()} ({train_mask.sum()/len(y)*100:.1f}%)")
    print(f"   ✓ 验证集: {val_mask.sum()} ({val_mask.sum()/len(y)*100:.1f}%)")
    print(f"   ✓ 测试集: {test_mask.sum()} ({test_mask.sum()/len(y)*100:.1f}%)")
    print(f"   ✓ 离职率: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
    
    return X, y, train_mask, val_mask, test_mask


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


def plot_curves(y_true, y_pred_proba, save_dir):
    """绘制PR曲线和ROC曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    aupr = average_precision_score(y_true, y_pred_proba)
    
    axes[0].plot(recall, precision, 'b-', linewidth=2, label=f'XGBoost (AUPR={aupr:.3f})')
    axes[0].plot([0, 1], [y_true.sum()/len(y_true)]*2, 'r--', label='Random')
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'XGBoost (AUROC={auroc:.3f})')
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
    plt.title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ 混淆矩阵已保存: {save_dir / 'confusion_matrix.png'}")


def plot_feature_importance(model, save_dir, top_n=20):
    """绘制特征重要性"""
    importance_dict = model.get_score(importance_type='gain')
    
    # 转换为DataFrame并排序
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in importance_dict.items()
    ]).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'].values)
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ 特征重要性已保存: {save_dir / 'feature_importance.png'}")
    
    return importance_df


def train_xgboost(X_train, y_train, X_val, y_val, args):
    """训练XGBoost模型"""
    print("\n2. 训练XGBoost模型...")
    
    # 计算pos_weight
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / num_pos
    
    print(f"   ✓ 正样本: {num_pos} ({num_pos/len(y_train)*100:.1f}%)")
    print(f"   ✓ 负样本: {num_neg} ({num_neg/len(y_train)*100:.1f}%)")
    print(f"   ✓ scale_pos_weight: {scale_pos_weight:.2f}")
    
    # XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',  # 使用AUPR作为评估指标
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'min_child_weight': args.min_child_weight,
        'scale_pos_weight': scale_pos_weight,
        'random_state': args.seed,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    print(f"\n   超参数:")
    for k, v in params.items():
        print(f"      {k}: {v}")
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # 训练
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    print("\n   开始训练...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=evals,
        early_stopping_rounds=args.patience,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    print(f"   ✓ 训练完成! 最佳迭代: {model.best_iteration}")
    print(f"   ✓ 最佳验证AUPR: {model.best_score:.4f}")
    
    return model, evals_result


def evaluate_model(model, X, y, mask_name='test'):
    """评估模型"""
    dtest = xgb.DMatrix(X)
    y_pred_proba = model.predict(dtest)
    
    # 计算指标
    metrics = {
        'aupr': average_precision_score(y, y_pred_proba),
        'auroc': roc_auc_score(y, y_pred_proba)
    }
    
    print(f"\n   {mask_name.upper()}集性能 (概率预测):")
    print(f"      AUPR:  {metrics['aupr']:.4f}")
    print(f"      AUROC: {metrics['auroc']:.4f}")
    
    return y_pred_proba, metrics


def main():
    parser = argparse.ArgumentParser(description='XGBoost Baseline训练')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, 
                       default='data/processed/homo_graph.pt',
                       help='数据路径')
    
    # XGBoost超参数
    parser.add_argument('--max-depth', type=int, default=6,
                       help='树的最大深度')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='学习率')
    parser.add_argument('--n-estimators', type=int, default=200,
                       help='树的数量')
    parser.add_argument('--subsample', type=float, default=0.8,
                       help='样本采样比例')
    parser.add_argument('--colsample-bytree', type=float, default=0.8,
                       help='特征采样比例')
    parser.add_argument('--min-child-weight', type=int, default=1,
                       help='最小子节点权重')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停patience')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save-dir', type=str, 
                       default='outputs/xgboost_baseline',
                       help='保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print("\n" + "="*70)
    print("📋 XGBoost Baseline 训练配置")
    print("="*70)
    print(f"\n数据: {args.data_path}")
    print(f"超参数:")
    print(f"   max_depth: {args.max_depth}")
    print(f"   learning_rate: {args.learning_rate}")
    print(f"   n_estimators: {args.n_estimators}")
    print(f"   subsample: {args.subsample}")
    print(f"   colsample_bytree: {args.colsample_bytree}")
    print(f"   patience: {args.patience}")
    print(f"随机种子: {args.seed}")
    print(f"保存目录: {args.save_dir}")
    print("="*70)
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    X, y, train_mask, val_mask, test_mask = load_data(args.data_path)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # 训练模型
    model, evals_result = train_xgboost(X_train, y_train, X_val, y_val, args)
    
    # 保存模型
    model_path = save_dir / 'xgboost_model.json'
    model.save_model(str(model_path))
    print(f"\n   ✓ 模型已保存: {model_path}")
    
    # 在验证集上找最优阈值
    print("\n3. 在验证集上寻找最优阈值...")
    val_pred_proba, _ = evaluate_model(model, X_val, y_val, 'val')
    best_threshold, best_f1, threshold_results = find_best_threshold(y_val, val_pred_proba)
    print(f"   ✓ 最优阈值: {best_threshold:.2f} (Val F1={best_f1:.4f})")
    
    # 在测试集上评估
    print("\n4. 在测试集上评估 (使用最优阈值)...")
    test_pred_proba, test_metrics_base = evaluate_model(model, X_test, y_test, 'test')
    
    # 使用最优阈值计算分类指标
    test_metrics = compute_metrics(y_test, test_pred_proba, best_threshold)
    
    print(f"\n   TEST集性能 (阈值={best_threshold:.2f}):")
    print(f"      AUPR:      {test_metrics['aupr']:.4f}")
    print(f"      AUROC:     {test_metrics['auroc']:.4f}")
    print(f"      F1:        {test_metrics['f1']:.4f}")
    print(f"      Precision: {test_metrics['precision']:.4f}")
    print(f"      Recall:    {test_metrics['recall']:.4f}")
    
    # 生成可视化
    print("\n5. 生成可视化...")
    plot_curves(y_test, test_pred_proba, save_dir)
    
    test_pred = (test_pred_proba >= best_threshold).astype(int)
    plot_confusion_matrix(y_test, test_pred, save_dir)
    
    importance_df = plot_feature_importance(model, save_dir)
    
    # 保存结果
    print("\n6. 保存结果...")
    results = {
        'model': 'XGBoost',
        'seed': args.seed,
        'hyperparameters': {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'min_child_weight': args.min_child_weight,
            'patience': args.patience
        },
        'best_iteration': int(model.best_iteration),
        'best_val_aupr': float(model.best_score),
        'best_threshold': float(best_threshold),
        'best_val_f1': float(best_f1),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'threshold_scan': threshold_results,
        'training_history': {
            'train_aupr': [float(x) for x in evals_result['train']['aucpr']],
            'val_aupr': [float(x) for x in evals_result['val']['aucpr']]
        },
        'feature_importance': importance_df.to_dict('records')
    }
    
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ 结果已保存: {results_path}")
    
    # 保存特征重要性CSV
    importance_csv_path = save_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"   ✓ 特征重要性已保存: {importance_csv_path}")
    
    # 生成文本报告
    report_path = save_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("XGBoost Baseline - 评估报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. 模型配置\n")
        f.write(f"   - 模型: XGBoost\n")
        f.write(f"   - 随机种子: {args.seed}\n")
        f.write(f"   - 最大深度: {args.max_depth}\n")
        f.write(f"   - 学习率: {args.learning_rate}\n")
        f.write(f"   - 树的数量: {args.n_estimators}\n")
        f.write(f"   - 最佳迭代: {model.best_iteration}\n\n")
        
        f.write("2. 数据集信息\n")
        f.write(f"   - 训练集: {train_mask.sum()} 样本\n")
        f.write(f"   - 验证集: {val_mask.sum()} 样本\n")
        f.write(f"   - 测试集: {test_mask.sum()} 样本\n")
        f.write(f"   - 离职率: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)\n\n")
        
        f.write("3. 验证集性能\n")
        f.write(f"   - 最佳AUPR: {model.best_score:.4f}\n")
        f.write(f"   - 最优阈值: {best_threshold:.2f}\n")
        f.write(f"   - 最优F1: {best_f1:.4f}\n\n")
        
        f.write("4. 测试集性能\n")
        f.write(f"   - AUPR:      {test_metrics['aupr']:.4f}\n")
        f.write(f"   - AUROC:     {test_metrics['auroc']:.4f}\n")
        f.write(f"   - F1 Score:  {test_metrics['f1']:.4f}\n")
        f.write(f"   - Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"   - Recall:    {test_metrics['recall']:.4f}\n\n")
        
        f.write("5. Top 10特征重要性\n")
        for i, row in importance_df.head(10).iterrows():
            f.write(f"   {i+1}. {row['feature']}: {row['importance']:.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"   ✓ 评估报告已保存: {report_path}")
    
    print("\n" + "="*70)
    print("✅ XGBoost Baseline 训练完成!")
    print("="*70)
    print(f"\n📊 测试集性能总结:")
    print(f"   AUPR:      {test_metrics['aupr']:.4f}")
    print(f"   AUROC:     {test_metrics['auroc']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"\n📁 所有结果已保存至: {save_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()