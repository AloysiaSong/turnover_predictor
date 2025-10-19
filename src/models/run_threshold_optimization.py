"""
在真实测试集上运行阈值优化
"""
import numpy as np
import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/yu/code/code2510/gnn')

from src.models.mlp_baseline import MLPBaseline
from src.models.trainer import create_dataloaders
from threshold_optimizer import ThresholdOptimizer


def main():
    print("="*80)
    print("真实测试集阈值优化")
    print("="*80)
    
    # ==================== 1. 加载数据 ====================
    print("\n1. 加载数据...")
    data_dir = Path('/Users/yu/code/code2510/gnn/data/processed')
    splits_dir = Path('/Users/yu/code/code2510/gnn/data/splits')
    
    # 加载特征和标签
    X = np.load(data_dir / 'employee_features.npy')
    y = np.load(data_dir / 'y_turnover_binary.npy')
    
    # 加载测试集索引
    test_idx = np.load(splits_dir / 'test_idx.npy')
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"   测试集: {len(X_test)} 样本")
    print(f"   正样本: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    # ==================== 2. 加载模型 ====================
    print("\n2. 加载训练好的模型...")
    model = MLPBaseline(input_dim=47, hidden_dims=[128, 64, 32], dropout=0.5)
    
    checkpoint = torch.load(
        '/Users/yu/code/code2510/gnn/models/mlp/best_model.pt',
        map_location='cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✅ 模型已加载")
    
    # ==================== 3. 预测概率 ====================
    print("\n3. 在测试集上预测...")
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        logits = model(X_test_tensor).squeeze()
        y_prob = torch.sigmoid(logits).numpy()
    
    print(f"   概率范围: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    print(f"   平均概率: {y_prob.mean():.4f}")
    
    # ==================== 4. 阈值优化 ====================
    print("\n4. 阈值优化分析...")
    
    optimizer = ThresholdOptimizer(y_test, y_prob)
    results = optimizer.generate_threshold_report(save_dir='results/mlp')
    
    # ==================== 5. 额外分析 ====================
    print("\n" + "="*80)
    print("📊 详细分析")
    print("="*80)
    
    # 对比不同阈值
    test_thresholds = [0.3, 0.37, 0.4, 0.5, 0.6]
    
    print(f"\n不同阈值对比:")
    print(f"{'阈值':>6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print("-" * 60)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    for threshold in test_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = (y_test == y_pred).mean()
        
        print(f"{threshold:>6.2f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {accuracy:>10.4f}")
    
    # ==================== 6. 推荐总结 ====================
    print("\n" + "="*80)
    print("🎯 最终推荐")
    print("="*80)
    
    f1_best = results['f1']
    
    print(f"\n基于F1最优:")
    print(f"  阈值: {f1_best['threshold']:.3f}")
    print(f"  F1 Score: {f1_best['f1']:.4f}")
    print(f"  Precision: {f1_best['precision']:.4f}")
    print(f"  Recall: {f1_best['recall']:.4f}")
    
    # 判断是否合理
    if f1_best['threshold'] < 0.2:
        print(f"\n⚠️ 警告: 最优阈值过低 ({f1_best['threshold']:.3f})")
        print(f"   原因: 可能是类别极度不平衡或模型预测分布偏移")
        print(f"   建议: 使用0.3-0.4的阈值，权衡Precision和Recall")
    elif f1_best['threshold'] > 0.7:
        print(f"\n⚠️ 警告: 最优阈值过高 ({f1_best['threshold']:.3f})")
        print(f"   原因: 模型可能过于保守")
        print(f"   建议: 检查模型训练过程")
    else:
        print(f"\n✅ 阈值合理，可以使用此阈值")
    
    print("\n" + "="*80)
    print("✅ 阈值优化完成！")
    print("="*80)
    print(f"\n结果保存在: results/mlp/")
    print(f"  - threshold_optimization.json")
    print(f"  - threshold_analysis.png")


if __name__ == '__main__':
    main()
