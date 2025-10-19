"""
对比不同深度的GCN架构
shallow, default, deep, very_deep
"""

import torch
import sys
sys.path.insert(0, 'src/models')

from gcn import create_gcn_model
from trainer import quick_train
from evaluator import GCNEvaluator
import json


def train_and_evaluate(architecture, data, device='cpu'):
    """训练并评估指定架构"""
    print("\n" + "="*70)
    print(f"训练架构: {architecture.upper()}")
    print("="*70)
    
    # 创建模型
    model = create_gcn_model(
        in_channels=data.num_node_features,
        architecture=architecture,
        dropout=0.5
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {num_params:,}")
    
    # 训练
    trainer, history = quick_train(
        model=model,
        data=data,
        epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        early_stopping_patience=20,
        device=device,
        save_dir=f'outputs/models_{architecture}'
    )
    
    # 评估
    evaluator = GCNEvaluator(model, data, device=device)
    results = evaluator.full_evaluation(
        save_dir=f'outputs/evaluation_{architecture}'
    )
    
    # 训练曲线
    evaluator.plot_training_curves(
        history,
        save_path=f'outputs/evaluation_{architecture}/training_curves.png'
    )
    
    return {
        'architecture': architecture,
        'num_params': num_params,
        'best_epoch': trainer.best_epoch + 1,
        'best_val_f1': max(history['val_f1']),
        'test_acc': results['test']['accuracy'],
        'test_precision': results['test']['precision'],
        'test_recall': results['test']['recall'],
        'test_f1': results['test']['f1'],
        'test_auc_roc': results['test']['roc_auc'],
        'test_auc_pr': results['test']['pr_auc']
    }


print("\n" + "="*70)
print("🏗️ GCN架构对比实验")
print("="*70)

# 加载数据
print("\n加载数据...")
data = torch.load('data/processed/homo_graph.pt')
print(f"✓ 节点: {data.num_nodes}, 边: {data.num_edges}")

# 测试所有架构
architectures = ['shallow', 'default', 'deep', 'very_deep']
all_results = []

for arch in architectures:
    result = train_and_evaluate(arch, data, device='cpu')
    all_results.append(result)

# 对比总结
print("\n" + "="*70)
print("📊 架构对比总结")
print("="*70)

print(f"\n{'架构':<12} {'参数量':<10} {'Best Epoch':<12} {'Test F1':<10} {'Test Recall':<12} {'排名'}")
print("="*70)

# 按F1排序
sorted_results = sorted(all_results, key=lambda x: x['test_f1'], reverse=True)

for rank, result in enumerate(sorted_results, 1):
    marker = "⭐" if rank == 1 else ""
    print(f"{result['architecture']:<12} {result['num_params']:<10,} "
          f"{result['best_epoch']:<12} {result['test_f1']:<10.4f} "
          f"{result['test_recall']:<12.4f} #{rank} {marker}")

print("="*70)

# 详细对比表
print(f"\n详细指标对比:")
print("="*90)
print(f"{'架构':<12} {'F1':<8} {'Recall':<10} {'Precision':<12} {'AUC-ROC':<10} {'AUC-PR':<10}")
print("="*90)

for result in sorted_results:
    print(f"{result['architecture']:<12} {result['test_f1']:<8.4f} "
          f"{result['test_recall']:<10.4f} {result['test_precision']:<12.4f} "
          f"{result['test_auc_roc']:<10.4f} {result['test_auc_pr']:<10.4f}")

print("="*90)

# 保存对比结果
comparison_file = 'outputs/architecture_comparison.json'
with open(comparison_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ 对比结果已保存: {comparison_file}")

# 推荐
best = sorted_results[0]
print(f"\n🏆 推荐架构: {best['architecture'].upper()}")
print(f"   F1-Score:  {best['test_f1']:.4f}")
print(f"   Recall:    {best['test_recall']:.4f}")
print(f"   参数量:    {best['num_params']:,}")

print("\n" + "="*70)
print("✅ 架构对比完成！")
print("="*70)
