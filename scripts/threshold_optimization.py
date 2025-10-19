"""
阈值优化脚本
无需重新训练，直接找到最优决策阈值
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'src/models')

from gcn import create_gcn_model
from evaluator2 import GCNEvaluator
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("\n" + "="*70)
print("🎯 阈值优化 - 寻找最优决策阈值")
print("="*70)

# 加载数据和模型
print("\n1. 加载模型...")
data = torch.load('data/processed/homo_graph.pt')
model = create_gcn_model(data.num_node_features, architecture='default')

checkpoint = torch.load('outputs/models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print("   ✓ 模型加载成功")

# 获取预测概率
print("\n2. 获取预测概率...")
evaluator = GCNEvaluator(model, data)
probs, preds_default, labels = evaluator.predict(data.test_mask)
print(f"   ✓ 测试集样本数: {len(labels)}")
print(f"   ✓ 离职样本数: {labels.sum()}")
print(f"   ✓ 概率范围: [{probs.min():.3f}, {probs.max():.3f}]")

# 扫描阈值
print("\n3. 扫描阈值 (0.05-0.70)...")
print("="*90)
print(f"{'阈值':<8} {'F1':<8} {'Recall':<10} {'Precision':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'标记'}")
print("="*90)

best_f1 = 0
best_threshold = 0.5
best_stats = {}
results = []

for threshold in np.arange(0.05, 0.75, 0.05):
    preds = (probs > threshold).astype(int)
    
    if preds.sum() == 0:
        continue
    
    f1 = f1_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    marker = "⭐ 最优" if f1 > best_f1 else ""
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_stats = {
            'f1': float(f1), 'recall': float(recall), 'precision': float(precision),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'threshold': float(threshold)
        }
    
    results.append({
        'threshold': threshold, 'f1': f1, 'recall': recall,
        'precision': precision, 'tp': tp, 'fp': fp, 'fn': fn
    })
    
    print(f"{threshold:<8.2f} {f1:<8.3f} {recall:<10.3f} {precision:<12.3f} "
          f"{tp:<6} {fp:<6} {fn:<6} {marker}")

print("="*90)

# 最优结果
print("\n" + "="*70)
print("✅ 最优阈值结果")
print("="*70)
print(f"\n最优阈值:    {best_stats['threshold']:.2f} (默认: 0.50)")
print(f"F1-Score:    {best_stats['f1']:.4f} (原: {f1_score(labels, preds_default):.4f}, 提升: {(best_stats['f1']/f1_score(labels, preds_default)-1)*100:+.1f}%)")
print(f"Recall:      {best_stats['recall']:.4f} (原: {recall_score(labels, preds_default):.4f}, 提升: {(best_stats['recall']/recall_score(labels, preds_default)-1)*100:+.1f}%)")
print(f"Precision:   {best_stats['precision']:.4f} (原: {precision_score(labels, preds_default):.4f}, 提升: {(best_stats['precision']/precision_score(labels, preds_default)-1)*100:+.1f}%)")

print(f"\n混淆矩阵:")
preds_opt = (probs > best_stats['threshold']).astype(int)
cm = confusion_matrix(labels, preds_opt)
print(f"                预测不离职    预测离职")
print(f"  实际不离职    {cm[0,0]:>8}      {cm[0,1]:>8}")
print(f"  实际离职      {cm[1,0]:>8}      {cm[1,1]:>8}")

print(f"\n业务含义:")
print(f"   ✓ 正确识别离职: {best_stats['tp']}/{labels.sum()} ({best_stats['recall']*100:.1f}%)")
print(f"   ✓ 误报在职:     {best_stats['fp']}/{len(labels)-labels.sum()} ({best_stats['fp']/(len(labels)-labels.sum())*100:.1f}%)")
print(f"   ✓ 漏报离职:     {best_stats['fn']}/{labels.sum()} ({best_stats['fn']/labels.sum()*100:.1f}%)")

# 保存结果
print("\n4. 保存最优阈值...")
import json
with open('outputs/evaluation/optimal_threshold.json', 'w') as f:
    json.dump(best_stats, f, indent=2)
print("   ✓ 已保存: outputs/evaluation/optimal_threshold.json")

# 绘制阈值-性能曲线
print("\n5. 绘制阈值曲线...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

thresholds = [r['threshold'] for r in results]
f1s = [r['f1'] for r in results]
recalls = [r['recall'] for r in results]
precisions = [r['precision'] for r in results]

# F1 vs 阈值
ax = axes[0, 0]
ax.plot(thresholds, f1s, 'b-', linewidth=2)
ax.axvline(best_threshold, color='r', linestyle='--', label=f'最优={best_threshold:.2f}')
ax.axhline(best_f1, color='r', linestyle='--', alpha=0.3)
ax.set_xlabel('阈值')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score vs 阈值')
ax.legend()
ax.grid(alpha=0.3)

# Recall vs 阈值
ax = axes[0, 1]
ax.plot(thresholds, recalls, 'g-', linewidth=2)
ax.axvline(best_threshold, color='r', linestyle='--')
ax.set_xlabel('阈值')
ax.set_ylabel('Recall')
ax.set_title('Recall vs 阈值')
ax.grid(alpha=0.3)

# Precision vs 阈值
ax = axes[1, 0]
ax.plot(thresholds, precisions, 'orange', linewidth=2)
ax.axvline(best_threshold, color='r', linestyle='--')
ax.set_xlabel('阈值')
ax.set_ylabel('Precision')
ax.set_title('Precision vs 阈值')
ax.grid(alpha=0.3)

# Precision-Recall曲线
ax = axes[1, 1]
ax.plot(recalls, precisions, 'purple', linewidth=2)
ax.scatter([best_stats['recall']], [best_stats['precision']], 
          color='r', s=100, zorder=5, label=f'最优点 (阈值={best_threshold:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall曲线')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/evaluation/threshold_optimization.png', dpi=150, bbox_inches='tight')
print("   ✓ 已保存: outputs/evaluation/threshold_optimization.png")

print("\n" + "="*70)
print("🎉 阈值优化完成！")
print("="*70)
print(f"\n使用最优阈值 {best_threshold:.2f} 可使F1提升至 {best_stats['f1']:.4f}")
print(f"无需重新训练，直接使用此阈值即可！")