"""
SMOTE数据增强
通过合成新样本来平衡训练集
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.insert(0, 'src/models')

from gcn import create_gcn_model
from trainer2 import quick_train
from evaluator2 import GCNEvaluator


def smote_oversample(features, labels, train_mask, k=5, target_ratio=0.3):
    """
    SMOTE过采样
    
    Args:
        features: 节点特征
        labels: 标签
        train_mask: 训练集mask
        k: 最近邻数量
        target_ratio: 目标正样本比例
        
    Returns:
        增强后的features, labels, train_mask
    """
    print("\n执行SMOTE数据增强...")
    
    # 获取训练集中的正负样本
    train_features = features[train_mask].numpy()
    train_labels = labels[train_mask].numpy()
    
    pos_indices = np.where(train_labels == 1)[0]
    neg_indices = np.where(train_labels == 0)[0]
    
    num_pos = len(pos_indices)
    num_neg = len(neg_indices)
    num_train = len(train_labels)
    
    print(f"   原始正样本: {num_pos} ({num_pos/num_train:.1%})")
    print(f"   原始负样本: {num_neg} ({num_neg/num_train:.1%})")
    
    # 计算需要合成的样本数
    num_needed = int(num_train * target_ratio / (1 - target_ratio)) - num_pos
    
    if num_needed <= 0:
        print(f"   无需增强，已达到目标比例")
        return features, labels, train_mask
    
    print(f"   需要合成: {num_needed} 个正样本")
    
    # 使用KNN找最近邻
    pos_features = train_features[pos_indices]
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(pos_features))).fit(pos_features)
    
    # 合成新样本
    synthetic_features = []
    for _ in range(num_needed):
        # 随机选择一个正样本
        idx = np.random.randint(0, len(pos_features))
        sample = pos_features[idx]
        
        # 找到k个最近邻
        distances, indices = nbrs.kneighbors([sample])
        # 随机选择一个邻居（排除自己）
        neighbor_idx = np.random.choice(indices[0][1:])
        neighbor = pos_features[neighbor_idx]
        
        # 在两个样本之间插值
        alpha = np.random.random()
        synthetic = sample + alpha * (neighbor - sample)
        synthetic_features.append(synthetic)
    
    synthetic_features = np.array(synthetic_features)
    
    # 合并原始数据和合成数据
    # 注意：合成的节点不在图中，所以他们的邻居信息会是随机的
    # 这是一个简化版本，完整版本需要重新构建图
    
    print(f"   ✓ 合成完成: {len(synthetic_features)} 个新样本")
    print(f"   新正样本比例: {(num_pos + num_needed)/(num_train + num_needed):.1%}")
    
    # 扩展特征矩阵
    features_extended = torch.cat([
        features,
        torch.from_numpy(synthetic_features).float()
    ], dim=0)
    
    # 扩展标签
    labels_extended = torch.cat([
        labels,
        torch.ones(len(synthetic_features), dtype=labels.dtype)
    ])
    
    # 扩展train_mask
    train_mask_extended = torch.cat([
        train_mask,
        torch.ones(len(synthetic_features), dtype=torch.bool)
    ])
    
    return features_extended, labels_extended, train_mask_extended


print("\n" + "="*70)
print("🔄 SMOTE数据增强训练")
print("="*70)

# 加载数据
print("\n1. 加载原始数据...")
data = torch.load('data/processed/homo_graph.pt')
print(f"   节点: {data.num_nodes}")

# SMOTE增强
features_aug, labels_aug, train_mask_aug = smote_oversample(
    data.x,
    data.y,
    data.train_mask,
    k=5,
    target_ratio=0.25  # 目标：25%正样本
)

# 创建增强后的数据对象
# 注意：这里我们保持原来的图结构，新节点会有随机的邻居
from torch_geometric.data import Data

# 扩展边索引（新节点的边）
num_new = len(labels_aug) - data.num_nodes
if num_new > 0:
    print(f"\n2. 为{num_new}个新节点创建边...")
    # 简化版本：每个新节点随机连接到k个原始节点
    k_connect = 10
    new_edges = []
    for i in range(num_new):
        new_node_idx = data.num_nodes + i
        # 随机选择k个原始节点连接
        targets = np.random.choice(data.num_nodes, k_connect, replace=False)
        for target in targets:
            new_edges.append([new_node_idx, target])
            new_edges.append([target, new_node_idx])  # 无向图
    
    new_edges = torch.tensor(new_edges, dtype=torch.long).t()
    edge_index_aug = torch.cat([data.edge_index, new_edges], dim=1)
    print(f"   ✓ 新增边: {new_edges.shape[1]}")
else:
    edge_index_aug = data.edge_index

# 创建增强数据对象
data_aug = Data(
    x=features_aug,
    edge_index=edge_index_aug,
    y=labels_aug,
    train_mask=train_mask_aug,
    val_mask=torch.cat([data.val_mask, torch.zeros(num_new, dtype=torch.bool)]),
    test_mask=torch.cat([data.test_mask, torch.zeros(num_new, dtype=torch.bool)])
)

print(f"\n增强后数据:")
print(f"   节点: {data_aug.num_nodes} (原: {data.num_nodes}, 新: {num_new})")
print(f"   边: {data_aug.num_edges} (原: {data.num_edges})")
print(f"   训练集: {data_aug.train_mask.sum()} (原: {data.train_mask.sum()})")

# 创建模型
print("\n3. 创建模型...")
model = create_gcn_model(
    in_channels=data_aug.num_node_features,
    architecture='default',
    dropout=0.5
)

# 训练
print("\n4. 训练模型...")
trainer, history = quick_train(
    model=model,
    data=data_aug,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    early_stopping_patience=20,
    device='cpu',
    save_dir='outputs/models_smote'
)

# 评估（只在原始测试集上）
print("\n5. 评估（原始测试集）...")
evaluator = GCNEvaluator(model, data_aug)

# 手动评估原始测试集
test_loss, test_acc, test_f1 = trainer.evaluate(data.test_mask)
print(f"\n测试集结果:")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   Accuracy:  {test_acc:.4f}")

# 完整评估
results = evaluator.full_evaluation(save_dir='outputs/evaluation_smote')

print("\n" + "="*70)
print("✅ SMOTE增强训练完成！")
print("="*70)
print(f"\n结果保存在: outputs/evaluation_smote/")
