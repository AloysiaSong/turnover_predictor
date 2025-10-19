"""
使用Focal Loss训练GCN
专门处理类别不平衡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, 'src/models')

from gcn import create_gcn_model
from trainer2 import GCNTrainer
from evaluator2 import GCNEvaluator


class FocalLoss(nn.Module):
    """
    Focal Loss: 更关注难分类样本
    
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    
    参数:
        alpha: 类别权重 (正样本权重)
        gamma: 聚焦参数 (越大越关注难样本)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # pt = P(y=1|x) if y=1 else P(y=0|x)
        pt = torch.exp(-BCE_loss)
        
        # Alpha权重
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        
        return F_loss.mean()


print("\n" + "="*70)
print("🔥 使用Focal Loss训练GCN")
print("="*70)

# 加载数据
print("\n1. 加载数据...")
data = torch.load('data/processed/homo_graph.pt')
print(f"   ✓ 节点: {data.num_nodes}, 边: {data.num_edges}")

# 创建模型
print("\n2. 创建模型...")
model = create_gcn_model(
    in_channels=data.num_node_features,
    architecture='default',
    dropout=0.5
)
print(f"   ✓ 参数量: {sum(p.numel() for p in model.parameters()):,}")

# 创建训练器（使用Focal Loss）
print("\n3. 创建训练器 (Focal Loss)...")
num_pos = data.y[data.train_mask].sum().item()
num_neg = data.train_mask.sum().item() - num_pos

# 计算alpha (正样本权重)
alpha = num_neg / (num_pos + num_neg)
print(f"   Alpha: {alpha:.3f} (根据类别比例自动计算)")
print(f"   Gamma: 2.0 (聚焦参数)")

trainer = GCNTrainer(
    model=model,
    data=data,
    device='cpu',
    lr=0.01,
    weight_decay=5e-4,
    pos_weight=None  # Focal Loss不需要pos_weight
)

# 替换损失函数为Focal Loss
trainer.criterion = FocalLoss(alpha=alpha, gamma=2.0)

print("\n4. 开始训练...")
history = trainer.train(
    epochs=200,
    early_stopping_patience=20,
    save_dir='outputs/models_focal',
    verbose=True
)

# 评估
print("\n5. 评估性能...")
test_loss, test_acc, test_f1 = trainer.evaluate(data.test_mask)
print(f"\n测试集快速评估:")
print(f"   Loss:     {test_loss:.4f}")
print(f"   Accuracy: {test_acc:.4f}")
print(f"   F1-Score: {test_f1:.4f}")

# 完整评估
print("\n6. 完整评估...")
evaluator = GCNEvaluator(model, data)
results = evaluator.full_evaluation(save_dir='outputs/evaluation_focal')

# 绘制训练曲线
evaluator.plot_training_curves(
    history,
    save_path='outputs/evaluation_focal/training_curves.png'
)

print("\n" + "="*70)
print("✅ Focal Loss训练完成！")
print("="*70)
print(f"\n最终结果:")
print(f"   F1-Score:  {results['test']['f1']:.4f}")
print(f"   Recall:    {results['test']['recall']:.4f}")
print(f"   Precision: {results['test']['precision']:.4f}")
print(f"   AUC-PR:    {results['test']['pr_auc']:.4f}")

print(f"\n结果保存在: outputs/evaluation_focal/")
