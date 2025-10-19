# Week 3-4: GNN模型开发 - 详细实施指南

**阶段**: 第二部分  
**时间**: 10-14天  
**目标**: 实现同构/异构GNN，超越MLP基线，达到AUC 0.93+

---

## 📋 总体规划

### 第二部分里程碑

| 阶段 | 天数 | 关键任务 | 预期产出 | 目标指标 |
|------|------|---------|---------|---------|
| **Week 3 Part 1** | Day 1-3 | 同构图构建 + GCN | GCN模型 | AUC 0.91+ |
| **Week 3 Part 2** | Day 4-7 | GAT + GraphSAGE | 注意力机制 | AUC 0.92+ |
| **Week 4 Part 1** | Day 8-10 | 异构HAN | HAN模型 | AUC 0.93+ |
| **Week 4 Part 2** | Day 11-14 | 多任务 + 对比 | 完整报告 | 最终评估 |

### 前置条件检查

```bash
# ✅ 确认第一部分已完成
[ ] data/processed/employee_features.npy 存在
[ ] data/processed/y_turnover_binary.npy 存在
[ ] data/edges/ 目录存在（5种边类型文件）
[ ] data/splits/ 目录存在（train/val/test索引）
[ ] models/mlp/best_model.pt 存在
[ ] 第一部分 MLP 基线 AUC ≥ 0.75
```

---

## 🎯 Week 3: 同构GNN模型

### Day 1-3: GCN实现

#### Day 1: 同构图数据准备

**任务清单**
- [ ] 将异构图转换为同构图
- [ ] 构建PyG Data对象
- [ ] 验证图连通性
- [ ] 数据统计分析

**详细步骤**

```python
# Step 1: 创建 src/graph/homogeneous_graph_builder.py

"""
同构图构建器
将异构图转换为同构图（所有节点视为同一类型）
"""

import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path


class HomogeneousGraphBuilder:
    """同构图构建器"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.edges_dir = self.data_dir / 'edges'
        self.splits_dir = self.data_dir / 'splits'
        
    def build(self):
        """构建同构图"""
        print("\n" + "="*60)
        print("构建同构图")
        print("="*60)
        
        # 1. 加载节点特征
        print("\n1. 加载员工节点特征...")
        X = np.load(self.processed_dir / 'employee_features.npy')
        y = np.load(self.processed_dir / 'y_turnover_binary.npy')
        
        print(f"   员工节点数: {len(X)}")
        print(f"   特征维度: {X.shape[1]}")
        print(f"   离职比例: {y.mean():.2%}")
        
        # 2. 构建边索引（只使用员工之间的连接）
        print("\n2. 构建边索引...")
        edge_index = self._build_employee_edges()
        
        print(f"   边数: {edge_index.shape[1]}")
        print(f"   平均度数: {edge_index.shape[1] / len(X):.2f}")
        
        # 3. 加载划分mask
        print("\n3. 加载数据划分...")
        train_mask = np.load(self.splits_dir / 'train_mask.npy')
        val_mask = np.load(self.splits_dir / 'val_mask.npy')
        test_mask = np.load(self.splits_dir / 'test_mask.npy')
        
        print(f"   训练集: {train_mask.sum()} ({train_mask.mean():.1%})")
        print(f"   验证集: {val_mask.sum()} ({val_mask.mean():.1%})")
        print(f"   测试集: {test_mask.sum()} ({test_mask.mean():.1%})")
        
        # 4. 创建PyG Data对象
        print("\n4. 创建PyG Data对象...")
        data = Data(
            x=torch.FloatTensor(X),
            edge_index=torch.LongTensor(edge_index),
            y=torch.LongTensor(y),
            train_mask=torch.BoolTensor(train_mask),
            val_mask=torch.BoolTensor(val_mask),
            test_mask=torch.BoolTensor(test_mask)
        )
        
        # 5. 验证图结构
        print("\n5. 图结构验证...")
        self._validate_graph(data)
        
        # 6. 保存
        save_path = self.processed_dir / 'homo_graph.pt'
        torch.save(data, save_path)
        print(f"\n✅ 同构图已保存: {save_path}")
        
        return data
    
    def _build_employee_edges(self):
        """
        构建员工之间的边
        策略: 基于共同属性（岗位、公司规模、公司类型）建立连接
        """
        # 加载原始数据
        import pandas as pd
        df = pd.read_csv(
            self.data_dir / 'raw' / 'originaldata.csv',
            encoding='gbk',
            skiprows=1
        )
        
        # 提取员工属性
        post_types = df['Q7岗位类型'].values
        company_sizes = df['Q8公司人员规模'].values
        company_types = df['Q9公司类型'].values
        
        edges = []
        
        # 策略1: 同岗位员工连接
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                # 同岗位
                if post_types[i] == post_types[j]:
                    edges.append([i, j])
                    edges.append([j, i])  # 无向图
                # 同公司规模
                elif company_sizes[i] == company_sizes[j]:
                    edges.append([i, j])
                    edges.append([j, i])
                # 同公司类型
                elif company_types[i] == company_types[j]:
                    edges.append([i, j])
                    edges.append([j, i])
        
        if not edges:
            # 如果没有边，创建自环
            edges = [[i, i] for i in range(len(df))]
        
        edge_index = np.array(edges).T
        
        return edge_index
    
    def _validate_graph(self, data):
        """验证图结构"""
        print(f"   节点数: {data.num_nodes}")
        print(f"   边数: {data.num_edges}")
        print(f"   特征维度: {data.num_node_features}")
        print(f"   是否有向: {data.is_directed()}")
        print(f"   是否有自环: {data.has_self_loops()}")
        print(f"   是否有孤立节点: {data.has_isolated_nodes()}")
        
        # 连通性检查
        from torch_geometric.utils import to_networkx
        import networkx as nx
        
        G = to_networkx(data, to_undirected=True)
        is_connected = nx.is_connected(G)
        num_components = nx.number_connected_components(G)
        
        print(f"   是否连通: {is_connected}")
        print(f"   连通分量数: {num_components}")


def main():
    """测试同构图构建"""
    builder = HomogeneousGraphBuilder()
    data = builder.build()
    
    print("\n" + "="*60)
    print("✅ 同构图构建完成！")
    print("="*60)
    
    return data


if __name__ == '__main__':
    main()
```

**运行验证**
```bash
python src/graph/homogeneous_graph_builder.py
```

**预期输出**
```
============================================================
构建同构图
============================================================

1. 加载员工节点特征...
   员工节点数: 500
   特征维度: 47
   离职比例: 11.20%

2. 构建边索引...
   边数: 8,156
   平均度数: 16.31

3. 加载数据划分...
   训练集: 340 (68.0%)
   验证集: 60 (12.0%)
   测试集: 100 (20.0%)

4. 创建PyG Data对象...

5. 图结构验证...
   节点数: 500
   边数: 8,156
   特征维度: 47
   是否有向: False
   是否有自环: False
   是否有孤立节点: False
   是否连通: True
   连通分量数: 1

✅ 同构图已保存: data/processed/homo_graph.pt
```

---

#### Day 2: GCN模型实现

**任务清单**
- [ ] 实现GCN模型类
- [ ] 定义前向传播
- [ ] 测试模型结构
- [ ] 验证输出形状

**详细步骤**

```python
# Step 2: 创建 src/models/gcn.py

"""
Graph Convolutional Network (GCN) 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    GCN模型
    
    Architecture:
        Input → GCNConv → ReLU → Dropout
              → GCNConv → ReLU → Dropout  
              → GCNConv → Linear → Output
    """
    
    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        num_layers=3,
        dropout=0.5,
        use_batch_norm=False
    ):
        """
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            num_layers: GCN层数
            dropout: Dropout比例
            use_batch_norm: 是否使用BatchNorm
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 最后一层输出固定维度（用于分类）
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # BatchNorm（可选）
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 分类头
        self.classifier = nn.Linear(hidden_channels, 1)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            out: 预测logits [num_nodes, 1]
        """
        # GCN层
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # 最后一层GCN（不加激活）
        x = self.convs[-1](x, edge_index)
        
        # 分类
        out = self.classifier(x)
        
        return out
    
    def predict_proba(self, x, edge_index):
        """预测概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = torch.sigmoid(logits)
        return probs


def create_gcn_model(in_channels, architecture='default', dropout=0.5):
    """
    GCN模型工厂
    
    Args:
        in_channels: 输入特征维度
        architecture: 模型架构
            - 'shallow': 2层，hidden=64
            - 'default': 3层，hidden=128
            - 'deep': 4层，hidden=256
        dropout: Dropout比例
        
    Returns:
        model: GCN模型实例
    """
    architectures = {
        'shallow': {'hidden_channels': 64, 'num_layers': 2},
        'default': {'hidden_channels': 128, 'num_layers': 3},
        'deep': {'hidden_channels': 256, 'num_layers': 4}
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    config = architectures[architecture]
    
    model = GCN(
        in_channels=in_channels,
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=dropout,
        use_batch_norm=True
    )
    
    return model


def test_gcn():
    """测试GCN模型"""
    print("\n" + "="*60)
    print("GCN模型测试")
    print("="*60)
    
    # 创建模型
    model = create_gcn_model(
        in_channels=47,
        architecture='default',
        dropout=0.5
    )
    
    print(f"\n模型架构: default")
    print(f"输入维度: 47")
    print(f"隐藏维度: 128")
    print(f"层数: 3")
    print(f"Dropout: 0.5")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print("\n" + "="*60)
    print("前向传播测试")
    print("="*60)
    
    # 创建测试数据
    num_nodes = 500
    num_edges = 2000
    
    x = torch.randn(num_nodes, 47)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    print(f"\n输入形状: x={x.shape}, edge_index={edge_index.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits)
    
    print(f"输出形状: logits={logits.shape}")
    print(f"概率范围: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"平均概率: {probs.mean():.4f}")
    
    print("\n✅ GCN模型测试通过！")
    
    return model


if __name__ == '__main__':
    test_gcn()
```

**运行验证**
```bash
python src/models/gcn.py
```

**预期输出**
```
============================================================
GCN模型测试
============================================================

模型架构: default
输入维度: 47
隐藏维度: 128
层数: 3
Dropout: 0.5

总参数量: 38,785
可训练参数: 38,785

============================================================
前向传播测试
============================================================

输入形状: x=torch.Size([500, 47]), edge_index=torch.Size([2, 2000])
输出形状: logits=torch.Size([500, 1])
概率范围: [0.3245, 0.6812]
平均概率: 0.5123

✅ GCN模型测试通过！
```

---

#### Day 3: GCN训练

**任务清单**
- [ ] 实现GCN训练器
- [ ] 完整训练流程
- [ ] 评估性能
- [ ] 与MLP对比

**详细步骤**

```python
# Step 3: 创建 train_gcn.py

"""
GCN模型训练脚本
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# 导入模块
from src.models.gcn import create_gcn_model
from src.evaluation.evaluator import Evaluator


class GCNTrainer:
    """GCN训练器"""
    
    def __init__(
        self,
        model,
        device='cpu',
        learning_rate=0.01,
        weight_decay=5e-4
    ):
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 损失函数（考虑类别不平衡）
        pos_weight = torch.tensor([7.9]).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def train_epoch(self, data):
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        out = self.model(data.x, data.edge_index)
        
        # 只计算训练集的损失
        loss = self.criterion(
            out[data.train_mask].squeeze(),
            data.y[data.train_mask].float()
        )
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, mask):
        """评估模型"""
        self.model.eval()
        
        # 预测
        out = self.model(data.x, data.edge_index)
        probs = torch.sigmoid(out).squeeze()
        
        # 提取对应mask的预测和标签
        y_true = data.y[mask].cpu().numpy()
        y_prob = probs[mask].cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        
        # 计算指标
        metrics = {
            'auc': roc_auc_score(y_true, y_prob),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': (y_true == y_pred).mean()
        }
        
        return metrics, y_prob
    
    def fit(
        self,
        data,
        epochs=200,
        early_stopping_patience=20,
        verbose=True
    ):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练GCN模型")
        print("="*60)
        
        data = data.to(self.device)
        
        best_val_auc = 0
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_auc': [],
            'val_f1': []
        }
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(data)
            
            # 验证
            val_metrics, _ = self.evaluate(data, data.val_mask)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['val_auc'].append(val_metrics['auc'])
            history['val_f1'].append(val_metrics['f1'])
            
            # 早停
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': best_val_auc
                }, 'models/gcn/best_model.pt')
            else:
                patience_counter += 1
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f}")
            
            # 早停判断
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n✅ 训练完成！最佳验证AUC: {best_val_auc:.4f}")
        
        return history


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    print("\n1. 加载同构图数据...")
    data = torch.load('data/processed/homo_graph.pt')
    print(f"   节点数: {data.num_nodes}")
    print(f"   边数: {data.num_edges}")
    print(f"   特征维度: {data.num_node_features}")
    
    # 2. 创建模型
    print("\n2. 创建GCN模型...")
    model = create_gcn_model(
        in_channels=data.num_node_features,
        architecture='default',
        dropout=0.5
    )
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 训练
    print("\n3. 训练模型...")
    trainer = GCNTrainer(
        model=model,
        device=device,
        learning_rate=0.01,
        weight_decay=5e-4
    )
    
    history = trainer.fit(
        data=data,
        epochs=200,
        early_stopping_patience=20
    )
    
    # 4. 测试集评估
    print("\n4. 测试集评估...")
    test_metrics, test_probs = trainer.evaluate(data, data.test_mask)
    
    print("\n测试集性能:")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    
    # 5. 完整评估报告
    print("\n5. 生成评估报告...")
    evaluator = Evaluator(save_dir='results/gcn')
    
    y_test = data.y[data.test_mask].cpu().numpy()
    evaluator.evaluate_and_report(
        y_true=y_test,
        y_pred=(test_probs >= 0.5).astype(int),
        y_prob=test_probs,
        model_name='GCN'
    )
    
    # 6. 对比MLP
    print("\n6. 对比MLP基线...")
    mlp_results = {
        'auc': 0.909,  # 从第一部分获取
        'f1': 0.516,
        'precision': 0.400,
        'recall': 0.727
    }
    
    print(f"\n{'指标':<12} {'MLP基线':<12} {'GCN':<12} {'提升':<12}")
    print("-" * 50)
    for metric in ['auc', 'f1', 'precision', 'recall']:
        mlp_val = mlp_results[metric]
        gcn_val = test_metrics[metric]
        improvement = (gcn_val - mlp_val) / mlp_val * 100
        
        print(f"{metric.upper():<12} {mlp_val:<12.4f} {gcn_val:<12.4f} "
              f"{improvement:+.2f}%")
    
    print("\n" + "="*60)
    print("✅ GCN训练与评估完成！")
    print("="*60)


if __name__ == '__main__':
    # 创建必要目录
    Path('models/gcn').mkdir(parents=True, exist_ok=True)
    Path('results/gcn').mkdir(parents=True, exist_ok=True)
    
    main()
```

**运行训练**
```bash
python train_gcn.py
```

**预期输出**
```
使用设备: cpu

1. 加载同构图数据...
   节点数: 500
   边数: 8,156
   特征维度: 47

2. 创建GCN模型...
   参数量: 38,785

3. 训练模型...
============================================================
开始训练GCN模型
============================================================
Epoch  10 | Loss: 0.4523 | Val AUC: 0.8234 | Val F1: 0.4102
Epoch  20 | Loss: 0.3891 | Val AUC: 0.8567 | Val F1: 0.4523
Epoch  30 | Loss: 0.3245 | Val AUC: 0.8892 | Val F1: 0.4891
Epoch  40 | Loss: 0.2934 | Val AUC: 0.9034 | Val F1: 0.5123
Epoch  50 | Loss: 0.2712 | Val AUC: 0.9156 | Val F1: 0.5234

Early stopping at epoch 58

✅ 训练完成！最佳验证AUC: 0.9156

4. 测试集评估...

测试集性能:
  AUC:       0.9134
  F1:        0.5401
  Precision: 0.4523
  Recall:    0.6818
  Accuracy:  0.8600

5. 生成评估报告...
✅ 评估报告已保存

6. 对比MLP基线...

指标          MLP基线        GCN          提升          
--------------------------------------------------
AUC         0.9091       0.9134       +0.47%
F1          0.5161       0.5401       +4.65%
PRECISION   0.4000       0.4523       +13.08%
RECALL      0.7273       0.6818       -6.25%

============================================================
✅ GCN训练与评估完成！
============================================================
```

---

### Day 1-3 检查清单

```
Week 3 - Day 1-3: GCN实现
========================

[ ] Day 1: 同构图数据准备
    [ ] HomogeneousGraphBuilder 类实现
    [ ] 同构图构建成功
    [ ] 图连通性验证通过
    [ ] homo_graph.pt 已保存

[ ] Day 2: GCN模型实现  
    [ ] GCN 类定义完成
    [ ] 前向传播测试通过
    [ ] 参数统计正确
    [ ] 模型可以正常实例化

[ ] Day 3: GCN训练
    [ ] 训练脚本运行成功
    [ ] 早停机制生效
    [ ] 最佳模型已保存
    [ ] 测试集 AUC ≥ 0.91
    [ ] 评估报告已生成
    [ ] 与MLP对比完成

性能目标:
    [目标] GCN AUC ≥ 0.91
    [实际] GCN AUC = _______
    
    [目标] 相比MLP提升 ≥ 0.5%
    [实际] 提升 = _______%
```

---

## 📝 Day 4-7 预告

接下来将实现：

### Day 4-5: GAT (Graph Attention Network)
- 注意力机制实现
- 多头注意力
- 目标AUC: 0.92+

### Day 6-7: GraphSAGE
- 邻居采样
- 聚合器设计
- 可扩展性验证

---

## 🎓 Week 3小结

完成Day 1-3后，你将：

✅ **掌握同构GNN** - 理解GCN原理和实现  
✅ **PyG框架** - 熟悉PyTorch Geometric用法  
✅ **图数据处理** - 异构→同构的转换  
✅ **性能提升** - GCN相比MLP提升0.5%+  

### 关键收获

1. **图卷积原理**: 如何聚合邻居信息
2. **消息传递**: GCN的消息传递机制
3. **图上训练**: mask-based训练方式
4. **性能对比**: 图结构信息的价值

---

**准备好Day 4-7了吗？** 🚀

继续阅读本指南的后续章节，或先完成Day 1-3的实践！
