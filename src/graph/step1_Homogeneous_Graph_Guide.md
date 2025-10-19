# Day 1: 同构图数据准备 - 完整指南

**目标**: 将异构图转换为同构图，为GCN模型做准备  
**时间**: 2-3小时  
**难度**: ⭐⭐

---

## 📋 任务清单

- [ ] 创建同构图构建器脚本
- [ ] 理解4种边构建策略
- [ ] 运行脚本并验证结果
- [ ] 检查图质量指标
- [ ] (可选) 可视化图结构

---

## 🎯 核心概念

### 什么是同构图？

**异构图** (Heterogeneous Graph):
- 多种类型的节点（员工、岗位、公司）
- 多种类型的边（工作关系、偏好关系）

**同构图** (Homogeneous Graph):
- 只有一种类型的节点（只看员工）
- 只有一种类型的边（员工之间的关系）

### 为什么需要转换？

1. **简化问题**: GCN等基础GNN只能处理同构图
2. **聚焦任务**: 离职预测只关心员工节点
3. **易于实现**: 同构图的PyG实现更简单
4. **性能基准**: 作为后续异构GNN的对比基准

---

## 🔧 Step 1: 创建脚本

### 1.1 创建目录结构

```bash
# 在项目根目录执行
mkdir -p src/graph
touch src/graph/__init__.py
```

### 1.2 复制代码

将 `homogeneous_graph_builder.py` 放到 `src/graph/` 目录：

```bash
cp homogeneous_graph_builder.py src/graph/
```

---

## 💡 Step 2: 理解边构建策略

### 策略1: 基于属性 (attribute)

**原理**: 具有相同属性的员工更可能有相似的行为模式

```python
# 连接规则
if same_post_type:      # 同岗位
    weight = 1.0
elif same_company_size: # 同公司规模
    weight = 0.7
elif same_company_type: # 同公司类型
    weight = 0.7
```

**优点**: 
- ✅ 符合业务直觉
- ✅ 计算快速
- ✅ 可解释性强

**缺点**:
- ⚠️ 可能产生孤立节点
- ⚠️ 忽略特征相似度

---

### 策略2: 基于相似度 (similarity)

**原理**: 特征相似的员工连接在一起

```python
# 计算余弦相似度
similarity = cosine_similarity(features)

# 连接相似度 > threshold 的节点
if similarity[i, j] > threshold:
    add_edge(i, j, weight=similarity[i, j])
```

**优点**:
- ✅ 捕捉特征空间的相似性
- ✅ 边权重有语义
- ✅ 全局视角

**缺点**:
- ⚠️ 计算量大 O(n²)
- ⚠️ 阈值难以选择
- ⚠️ 对特征缩放敏感

---

### 策略3: K近邻 (knn)

**原理**: 每个节点连接到其k个最近邻居

```python
# 基于欧氏距离找k近邻
knn_graph = kneighbors_graph(features, k=10)

# 距离转权重
weight = 1 / (1 + distance)
```

**优点**:
- ✅ 保证连通性
- ✅ 度数均匀
- ✅ 计算高效

**缺点**:
- ⚠️ k值需要调参
- ⚠️ 可能忽略重要的长距离连接
- ⚠️ 对异常值敏感

---

### 策略4: 混合 (hybrid) ⭐推荐

**原理**: 结合多种策略的优势

```python
# 1. 属性边（高权重）
attribute_edges × 1.5

# 2. k-NN边（确保连通性）
knn_edges × 1.0

# 3. 高相似度边（补充）
similarity_edges × 1.0

# 合并并去重（保留最大权重）
```

**优点**:
- ✅ 最全面
- ✅ 鲁棒性好
- ✅ 性能最佳

**缺点**:
- ⚠️ 计算稍慢
- ⚠️ 边数较多

---

## 🚀 Step 3: 运行脚本

### 3.1 基本运行

```bash
cd /Users/yu/code/code2510/gnn
python src/graph/homogeneous_graph_builder.py
```

### 3.2 预期输出

```
======================================================================
🚀 同构图构建脚本
======================================================================

======================================================================
🔨 同构图构建器
======================================================================

📂 Step 1/6: 加载数据...
   ✓ 员工节点数: 500
   ✓ 特征维度: 47
   ✓ 离职员工: 56 (11.20%)
   ✓ 在职员工: 444 (88.80%)

🔗 Step 2/6: 构建边 (策略=hybrid)...
   🔄 使用混合边策略...
   → 添加属性边...
   📋 使用属性边策略...
   ✓ 基于属性的边: 8124
   → 添加k-NN边 (k=10)...
   🎯 使用k-NN边策略 (k=10)...
   ✓ 基于k-NN的边: 5000
   → 添加高相似度边 (阈值=0.6)...
   📏 使用相似度边策略 (阈值=0.6)...
   ✓ 基于相似度的边: 3456
   ✓ 混合策略总边数: 12,458
   ✓ 边数: 12,458
   ✓ 平均度数: 24.92
   ✓ 边权重范围: [0.602, 1.500]

📊 Step 3/6: 加载数据划分...
   ✓ 训练集: 340 (68.0%)
   ✓ 验证集: 60 (12.0%)
   ✓ 测试集: 100 (20.0%)

🏗️ Step 4/6: 创建PyG Data对象...
   ✓ PyG Data对象创建完成
   ✓ 节点特征: torch.Size([500, 47])
   ✓ 边索引: torch.Size([2, 12458])
   ✓ 边权重: torch.Size([12458, 1])

✅ Step 5/6: 验证图结构...

----------------------------------------------------------------------
📊 图结构验证
----------------------------------------------------------------------

基本信息:
   节点数: 500
   边数: 12458
   特征维度: 47
   是否有向: False
   是否有自环: False
   是否有孤立节点: False

连通性分析:
   是否连通: True
   连通分量数: 1

度分布:
   最小度: 10
   最大度: 89
   平均度: 24.92
   中位数度: 22.00

边权重分布:
   最小权重: 0.602
   最大权重: 1.500
   平均权重: 1.023
   中位数权重: 1.000

数据划分:
   训练集: 340 节点
   验证集: 60 节点
   测试集: 100 节点

标签分布:
   训练集离职率: 11.18%
   验证集离职率: 11.67%
   测试集离职率: 11.00%

----------------------------------------------------------------------

💾 Step 6/6: 保存同构图...

📊 统计信息已保存: data/processed/homo_graph_stats.json

======================================================================
✅ 同构图构建完成！
📁 已保存: data/processed/homo_graph.pt
======================================================================

🧪 测试加载...
   ✅ 加载成功！节点数=500, 边数=12458

======================================================================
✅ 全部完成！
======================================================================

下一步: python src/models/gcn.py
```

---

## ✅ Step 4: 验证结果

### 4.1 检查文件是否生成

```bash
ls -lh data/processed/

# 应该看到:
# homo_graph.pt            # 同构图数据
# homo_graph_stats.json    # 统计信息
```

### 4.2 查看统计信息

```bash
cat data/processed/homo_graph_stats.json
```

```json
{
  "num_nodes": 500,
  "num_features": 47,
  "turnover_rate": 0.112,
  "num_edges": 12458,
  "avg_degree": 24.92,
  "strategy": "hybrid",
  "k": 10,
  "similarity_threshold": 0.6,
  "is_connected": true,
  "num_components": 1,
  "has_isolated_nodes": false,
  "min_degree": 10,
  "max_degree": 89
}
```

### 4.3 质量检查清单

```
图质量检查
==========

[ ] 节点数 = 500 ✅
[ ] 特征维度 = 47 ✅
[ ] 边数 > 5,000 ✅
[ ] 平均度数 > 10 ✅
[ ] 是否连通 = True ✅
[ ] 孤立节点 = False ✅
[ ] 最小度 ≥ 5 ✅
[ ] 训练/验证/测试离职率接近11% ✅
```

---

## 🎨 Step 5 (可选): 可视化

### 5.1 生成可视化

```python
# 在Python中运行
from src.graph.homogeneous_graph_builder import visualize_graph
import torch

data = torch.load('data/processed/homo_graph.pt')
visualize_graph(data, save_path='outputs/homo_graph_viz.png')
```

### 5.2 查看结果

```bash
open outputs/homo_graph_viz.png
# 或
xdg-open outputs/homo_graph_viz.png  # Linux
```

---

## 🔧 Step 6: 调优参数

### 如果图不连通

```python
# 增加k值
data = builder.build(strategy='hybrid', k=15)

# 或降低相似度阈值
data = builder.build(strategy='hybrid', similarity_threshold=0.5)
```

### 如果边太多（训练慢）

```python
# 减少k值
data = builder.build(strategy='hybrid', k=5)

# 或提高相似度阈值
data = builder.build(strategy='hybrid', similarity_threshold=0.7)

# 或只用knn策略
data = builder.build(strategy='knn', k=10)
```

### 如果想要更多语义连接

```python
# 只用属性策略
data = builder.build(strategy='attribute')

# 或降低相似度阈值
data = builder.build(strategy='hybrid', similarity_threshold=0.4)
```

---

## 🐛 常见问题

### Q1: ImportError: No module named 'torch_geometric'

**解决**:
```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Q2: 图不连通（num_components > 1）

**原因**: 边构建策略太严格

**解决**:
1. 使用 `strategy='hybrid'`
2. 增加 `k` 值（如 k=15）
3. 降低 `similarity_threshold`（如 0.5）

### Q3: 边数太少（< 3000）

**原因**: 参数设置过于严格

**解决**:
```python
data = builder.build(
    strategy='hybrid',
    k=15,                   # 增加k
    similarity_threshold=0.5  # 降低阈值
)
```

### Q4: 内存错误

**原因**: 相似度矩阵太大

**解决**:
```python
# 避免使用similarity策略
data = builder.build(strategy='knn', k=10)

# 或使用attribute策略
data = builder.build(strategy='attribute')
```

### Q5: 运行时间太长

**原因**: 相似度计算O(n²)复杂度

**解决**:
- 使用 `strategy='attribute'` 或 `'knn'`
- 减小相似度阈值（减少需要计算的边）

---

## 📊 性能基准

| 策略 | 边数 | 平均度 | 连通性 | 运行时间 |
|------|------|--------|--------|---------|
| attribute | ~8K | ~16 | 可能不连通 | 10s |
| similarity | ~5K | ~10 | 可能不连通 | 60s |
| knn | 5K | 10 | 必连通 | 15s |
| **hybrid** | **~12K** | **~25** | **必连通** | **30s** |

---

## ✅ Day 1 完成检查

```
Day 1 完成检查清单
==================

文件创建
[ ] src/graph/__init__.py
[ ] src/graph/homogeneous_graph_builder.py

文件生成
[ ] data/processed/homo_graph.pt
[ ] data/processed/homo_graph_stats.json

质量验证
[ ] 节点数 = 500
[ ] 边数 > 5,000
[ ] 图是连通的
[ ] 无孤立节点
[ ] 平均度数 > 10

理解检查
[ ] 理解4种边构建策略
[ ] 知道如何选择参数
[ ] 会查看统计信息
[ ] 会解决常见问题

如果以上全部✅，进入Day 2！
```

---

## 🎓 关键要点总结

1. **同构图vs异构图**: 同构图更简单，适合GCN等基础模型
2. **4种策略**: attribute(属性), similarity(相似度), knn(近邻), hybrid(混合)
3. **推荐配置**: `strategy='hybrid', k=10, threshold=0.6`
4. **质量指标**: 连通性、平均度数、边数
5. **下一步**: GCN模型实现（Day 2）

---

## 📚 扩展阅读

- PyTorch Geometric文档: https://pytorch-geometric.readthedocs.io/
- GCN原论文: "Semi-Supervised Classification with Graph Convolutional Networks"
- 图连通性: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)

---

**恭喜完成Day 1！准备好Day 2了吗？** 🎉

下一步: [Day 2: GCN模型实现](./Day2_GCN_Model.md)
