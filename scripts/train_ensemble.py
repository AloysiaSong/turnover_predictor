"""
GCN集成学习
训练多个模型，投票或平均预测
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'src/models')

from gcn import create_gcn_model
from trainer2 import GCNTrainer
from evaluator2 import GCNEvaluator
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import json


def train_single_model(data, model_id, config):
    """训练单个模型"""
    print(f"\n训练模型 #{model_id}...")
    
    model = create_gcn_model(
        in_channels=data.num_node_features,
        architecture=config['architecture'],
        dropout=config['dropout']
    )
    
    # 计算pos_weight
    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    pos_weight = num_neg / num_pos
    
    trainer = GCNTrainer(
        model=model,
        data=data,
        device='cpu',
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        pos_weight=pos_weight
    )
    
    history = trainer.train(
        epochs=config['epochs'],
        early_stopping_patience=config['patience'],
        save_dir=f"outputs/models_ensemble/model_{model_id}",
        verbose=False
    )
    
    # 评估
    test_loss, test_acc, test_f1 = trainer.evaluate(data.test_mask)
    
    print(f"   ✓ Test F1: {test_f1:.4f}, Epoch: {trainer.best_epoch+1}")
    
    return model, {
        'f1': test_f1,
        'acc': test_acc,
        'best_epoch': trainer.best_epoch + 1
    }


def ensemble_predict(models, data, method='vote'):
    """
    集成预测
    
    Args:
        models: 模型列表
        data: 数据
        method: 'vote' (投票) 或 'average' (平均概率)
    """
    all_probs = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(
                data.x,
                data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None
            )
            probs = torch.sigmoid(out[data.test_mask]).squeeze().cpu().numpy()
            all_probs.append(probs)
    
    all_probs = np.array(all_probs)  # (n_models, n_samples)
    
    if method == 'vote':
        # 投票法：每个模型预测0或1，多数投票
        votes = (all_probs > 0.5).astype(int)
        preds = (votes.sum(axis=0) > len(models) / 2).astype(int)
        avg_probs = all_probs.mean(axis=0)
    else:  # average
        # 平均法：平均所有模型的概率
        avg_probs = all_probs.mean(axis=0)
        preds = (avg_probs > 0.5).astype(int)
    
    labels = data.y[data.test_mask].cpu().numpy()
    
    return preds, avg_probs, labels


print("\n" + "="*70)
print("🎯 GCN集成学习")
print("="*70)

# 配置
config = {
    'architecture': 'default',
    'dropout': 0.5,
    'lr': 0.01,
    'weight_decay': 5e-4,
    'epochs': 200,
    'patience': 20
}

n_models = 5  # 训练5个模型

print(f"\n配置:")
for k, v in config.items():
    print(f"   {k}: {v}")
print(f"   n_models: {n_models}")

# 加载数据
print(f"\n加载数据...")
data = torch.load('data/processed/homo_graph.pt')
print(f"   ✓ 节点: {data.num_nodes}")

# 训练多个模型
print(f"\n{'='*70}")
print(f"训练 {n_models} 个模型")
print(f"{'='*70}")

models = []
individual_results = []

for i in range(n_models):
    model, result = train_single_model(data, i+1, config)
    models.append(model)
    individual_results.append(result)

# 单个模型性能
print(f"\n单个模型性能:")
print(f"{'模型':<10} {'F1':<10} {'Accuracy':<12} {'Best Epoch'}")
print("="*50)
for i, result in enumerate(individual_results, 1):
    print(f"Model {i:<4} {result['f1']:<10.4f} {result['acc']:<12.4f} {result['best_epoch']}")

print("="*50)
print(f"平均:      {np.mean([r['f1'] for r in individual_results]):.4f}")
print(f"最好:      {max([r['f1'] for r in individual_results]):.4f}")
print(f"最差:      {min([r['f1'] for r in individual_results]):.4f}")

# 集成预测
print(f"\n{'='*70}")
print("集成预测")
print(f"{'='*70}")

methods = ['vote', 'average']
ensemble_results = {}

for method in methods:
    print(f"\n方法: {method.upper()}")
    preds, probs, labels = ensemble_predict(models, data, method=method)
    
    f1 = f1_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   Precision: {precision:.4f}")
    
    cm = confusion_matrix(labels, preds)
    print(f"\n   混淆矩阵:")
    print(f"                预测不离职    预测离职")
    print(f"   实际不离职    {cm[0,0]:>8}      {cm[0,1]:>8}")
    print(f"   实际离职      {cm[1,0]:>8}      {cm[1,1]:>8}")
    
    ensemble_results[method] = {
        'f1': float(f1),
        'recall': float(recall),
        'precision': float(precision)
    }

# 对比总结
print(f"\n{'='*70}")
print("📊 集成 vs 单模型对比")
print(f"{'='*70}")

print(f"\n{'方法':<20} {'F1':<10} {'Recall':<12} {'Precision':<12} {'提升'}")
print("="*70)

# 最好的单模型
best_single_f1 = max([r['f1'] for r in individual_results])
print(f"{'最好单模型':<20} {best_single_f1:<10.4f} {'':12} {'':12} {'基线'}")

# 集成方法
for method, result in ensemble_results.items():
    improvement = (result['f1'] - best_single_f1) / best_single_f1 * 100
    marker = "⭐" if result['f1'] > best_single_f1 else ""
    print(f"{'集成-' + method:<20} {result['f1']:<10.4f} {result['recall']:<12.4f} "
          f"{result['precision']:<12.4f} {improvement:+.1f}% {marker}")

print("="*70)

# 保存结果
results_summary = {
    'config': config,
    'n_models': n_models,
    'individual_results': individual_results,
    'ensemble_results': ensemble_results,
    'best_single_f1': float(best_single_f1),
    'best_ensemble_f1': float(max([r['f1'] for r in ensemble_results.values()]))
}

with open('outputs/ensemble_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ 结果已保存: outputs/ensemble_results.json")

# 推荐
best_ensemble_method = max(ensemble_results, key=lambda k: ensemble_results[k]['f1'])
best_ensemble_f1 = ensemble_results[best_ensemble_method]['f1']

print(f"\n🏆 推荐方法: 集成-{best_ensemble_method.upper()}")
print(f"   F1-Score: {best_ensemble_f1:.4f}")
print(f"   提升: {(best_ensemble_f1 - best_single_f1) / best_single_f1 * 100:+.1f}%")

print("\n" + "="*70)
print("✅ 集成学习完成！")
print("="*70)
