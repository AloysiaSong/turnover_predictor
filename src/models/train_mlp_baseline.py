"""
MLP基线模型完整训练脚本
端到端训练流程
"""
import numpy as np
import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.append('/Users/yu/code/code2510/gnn')

# 导入模型和工具
from mlp_baseline import create_mlp_model
from trainer import Trainer, create_dataloaders
from evaluator import Evaluator


def load_data():
    """加载所有数据"""
    print("="*80)
    print("加载数据")
    print("="*80)
    
    # 数据路径
    data_dir = Path('/Users/yu/code/code2510/gnn/data/processed')
    splits_dir = Path('/Users/yu/code/code2510/gnn/data/splits')
    
    # 1. 加载特征
    X = np.load(data_dir / 'employee_features.npy')
    print(f"✅ 特征: {X.shape}")
    
    # 2. 加载标签
    y_binary = np.load(data_dir / 'y_turnover_binary.npy')
    print(f"✅ 标签: {y_binary.shape}")
    print(f"   正样本: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
    
    # 3. 加载划分索引
    train_idx = np.load(splits_dir / 'train_idx.npy')
    val_idx = np.load(splits_dir / 'val_idx.npy')
    test_idx = np.load(splits_dir / 'test_idx.npy')
    
    print(f"✅ 数据划分:")
    print(f"   Train: {len(train_idx)} 样本")
    print(f"   Val:   {len(val_idx)} 样本")
    print(f"   Test:  {len(test_idx)} 样本")
    
    # 4. 划分数据
    X_train, y_train = X[train_idx], y_binary[train_idx]
    X_val, y_val = X[val_idx], y_binary[val_idx]
    X_test, y_test = X[test_idx], y_binary[test_idx]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    """主训练流程"""
    print("\n" + "="*80)
    print("MLP基线模型训练")
    print("="*80)
    
    # ==================== 1. 加载数据 ====================
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    # ==================== 2. 创建数据加载器 ====================
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=32,
        num_workers=0
    )
    
    # ==================== 3. 创建模型 ====================
    model = create_mlp_model(
        input_dim=47,
        architecture='default',  # [128, 64, 32]
        dropout=0.5
    )
    
    # ==================== 4. 创建训练器 ====================
    trainer = Trainer(
        model=model,
        device=None,  # 自动选择
        learning_rate=0.001,
        weight_decay=1e-4,
        pos_weight=7.9  # 处理类别不平衡
    )
    
    # ==================== 5. 训练模型 ====================
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_dir='models/mlp',
        verbose=True
    )
    
    # ==================== 6. 绘制训练历史 ====================
    evaluator = Evaluator(save_dir='results/mlp')
    evaluator.plot_training_history(history)
    
    # ==================== 7. 在验证集上评估 ====================
    print("\n" + "="*80)
    print("验证集评估")
    print("="*80)
    
    val_loss, (val_probs, val_preds, val_labels) = trainer.evaluate(
        val_loader, 
        return_predictions=True
    )
    
    val_metrics = evaluator.generate_report(
        y_true=val_labels,
        y_pred=val_preds,
        y_prob=val_probs,
        set_name='Validation'
    )
    
    # ==================== 8. 在测试集上评估 ====================
    print("\n" + "="*80)
    print("测试集评估")
    print("="*80)
    
    test_loss, (test_probs, test_preds, test_labels) = trainer.evaluate(
        test_loader,
        return_predictions=True
    )
    
    test_metrics = evaluator.generate_report(
        y_true=test_labels,
        y_pred=test_preds,
        y_prob=test_probs,
        set_name='Test'
    )
    
    # ==================== 9. 总结 ====================
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    
    print(f"\n最终性能:")
    print(f"{'':15s} {'Val':>10s} {'Test':>10s}")
    print("-" * 40)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        val_val = val_metrics[metric]
        test_val = test_metrics[metric]
        print(f"{metric:15s} {val_val:10.4f} {test_val:10.4f}")
    
    print(f"\n模型保存在: models/mlp/")
    print(f"结果保存在: results/mlp/")
    
    print("\n✅ 训练完成！")
    
    return model, history, val_metrics, test_metrics


if __name__ == '__main__':
    model, history, val_metrics, test_metrics = main()
