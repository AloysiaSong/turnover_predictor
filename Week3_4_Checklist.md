# Week 3-4: GNN模型开发 - 快速检查清单 ✅

**目标**: 实现同构/异构GNN，超越MLP基线  
**时间**: 10-14天  
**状态**: 🎯 待开始

---

## 📦 前置条件检查

在开始Week 3-4之前，请确认：

```bash
# ✅ 第一部分完成度检查
[ ] data/processed/employee_features.npy 存在
[ ] data/processed/y_turnover_binary.npy 存在  
[ ] data/edges/ 目录存在（5个.pt文件）
[ ] data/splits/ 目录存在（6个文件）
[ ] models/mlp/best_model.pt 存在
[ ] reports/Week_1-2_Summary.md 存在
[ ] MLP基线 AUC ≥ 0.75

# 如果以上有任何未完成，请先完成第一部分
```

---

## 🎯 Week 3: 同构GNN (Day 1-7)

### Day 1-3: GCN实现

#### Day 1: 同构图数据准备 (2-3小时)

**核心任务**
```bash
[ ] 创建 src/graph/homogeneous_graph_builder.py
[ ] 实现 HomogeneousGraphBuilder 类
[ ] 基于共同属性构建员工间边
[ ] 验证图连通性
[ ] 保存 data/processed/homo_graph.pt
```

**运行验证**
```bash
python src/graph/homogeneous_graph_builder.py

# 预期输出
✅ 员工节点数: 500
✅ 边数: ~8,000+
✅ 是否连通: True
✅ 同构图已保存
```

**检查清单**
- [ ] 文件 `src/graph/homogeneous_graph_builder.py` 已创建
- [ ] 类 `HomogeneousGraphBuilder` 实现完成
- [ ] 方法 `build()` 运行无错误
- [ ] 方法 `_build_employee_edges()` 构建边成功
- [ ] 方法 `_validate_graph()` 验证通过
- [ ] 文件 `data/processed/homo_graph.pt` 已生成
- [ ] 图是连通的（无孤立节点）
- [ ] 边数 > 5,000（确保有足够连接）

**故障排查**
```bash
# 问题1: 边数太少或为0
→ 检查 _build_employee_edges() 中的连接策略
→ 确认原始数据中有共同属性

# 问题2: 图不连通
→ 添加更多连接策略（如基于技能相似度）
→ 或添加必要的边保证连通性

# 问题3: 内存错误
→ 减少边的数量
→ 或使用更高效的边构建方法
```

---

#### Day 2: GCN模型实现 (2-3小时)

**核心任务**
```bash
[ ] 创建 src/models/gcn.py
[ ] 实现 GCN 类
    [ ] __init__: 定义GCN层、BatchNorm、分类头
    [ ] forward: 实现前向传播
    [ ] predict_proba: 概率预测
[ ] 实现 create_gcn_model 工厂函数
[ ] 测试模型结构
```

**运行验证**
```bash
python src/models/gcn.py

# 预期输出
✅ 模型架构: default (3层)
✅ 参数量: ~38,785
✅ 前向传播测试通过
✅ 输出形状正确
```

**检查清单**
- [ ] 文件 `src/models/gcn.py` 已创建
- [ ] 类 `GCN` 实现完成
- [ ] GCNConv 层正确使用
- [ ] BatchNorm 和 Dropout 已添加
- [ ] 分类头输出维度为1
- [ ] `create_gcn_model` 支持3种架构
- [ ] 模型测试函数 `test_gcn()` 运行成功
- [ ] 参数量统计正确
- [ ] 前向传播无报错

**架构配置**
```python
architectures = {
    'shallow': hidden=64, layers=2,    # ~16K参数
    'default': hidden=128, layers=3,   # ~38K参数  ✅ 推荐
    'deep': hidden=256, layers=4       # ~150K参数
}
```

---

#### Day 3: GCN训练与评估 (3-4小时)

**核心任务**
```bash
[ ] 创建 train_gcn.py
[ ] 实现 GCNTrainer 类
    [ ] train_epoch: 训练一轮
    [ ] evaluate: 评估性能
    [ ] fit: 完整训练流程
[ ] 运行完整训练
[ ] 评估测试集性能
[ ] 与MLP基线对比
```

**运行训练**
```bash
python train_gcn.py

# 预计时间: 10-30分钟
```

**检查清单**
- [ ] 文件 `train_gcn.py` 已创建
- [ ] 类 `GCNTrainer` 实现完成
- [ ] 训练循环运行无错误
- [ ] 早停机制生效
- [ ] 文件 `models/gcn/best_model.pt` 已保存
- [ ] 训练历史记录完整
- [ ] 测试集评估完成
- [ ] 评估报告已生成在 `results/gcn/`
- [ ] 与MLP对比表格输出

**性能目标**
```
GCN性能目标
===========
测试集 AUC:       ≥ 0.910   [实际: _____ ]
测试集 F1:        ≥ 0.530   [实际: _____ ]
相比MLP AUC提升:  ≥ +0.5%   [实际: _____ ]
相比MLP F1提升:   ≥ +3%     [实际: _____ ]

训练时间:         ≤ 30分钟   [实际: _____ ]
最佳epoch:        50-100     [实际: _____ ]
```

**如果性能不达标**
```bash
# 调优策略
1. 增加hidden_channels (128→256)
2. 调整learning_rate (0.01→0.005)
3. 增加weight_decay (5e-4→1e-3)
4. 使用不同的边构建策略
5. 添加更多GCN层
```

---

### Day 1-3 里程碑检查

```
Week 3 - Day 1-3 完成度
=======================

数据准备
[ ] homo_graph.pt 已生成并验证

模型实现  
[ ] GCN 类实现完成
[ ] 模型测试通过

训练评估
[ ] GCN训练完成
[ ] 性能达到目标
[ ] 评估报告生成
[ ] 与MLP对比完成

交付物
[ ] src/graph/homogeneous_graph_builder.py
[ ] src/models/gcn.py
[ ] train_gcn.py
[ ] models/gcn/best_model.pt
[ ] results/gcn/ (评估报告)

如果以上全部✅，继续Day 4-7！
```

---

## 🎯 Day 4-7: GAT + GraphSAGE

### Day 4-5: GAT实现 (4-6小时)

**核心任务**
```bash
[ ] 创建 src/models/gat.py
[ ] 实现 GAT 类
    [ ] 使用 GATConv 层
    [ ] 多头注意力机制
    [ ] 注意力权重可视化
[ ] 创建 train_gat.py
[ ] 训练GAT模型
[ ] 评估与对比
```

**检查清单**
- [ ] `src/models/gat.py` 实现完成
- [ ] GAT支持多头注意力（heads=8推荐）
- [ ] `train_gat.py` 运行成功
- [ ] 测试集 AUC ≥ 0.920
- [ ] 相比GCN有提升
- [ ] 注意力权重可视化（可选）

**性能目标**
```
GAT性能目标
===========
测试集 AUC:       ≥ 0.920   [实际: _____ ]
测试集 F1:        ≥ 0.550   [实际: _____ ]
相比GCN AUC提升:  ≥ +1%     [实际: _____ ]
```

---

### Day 6-7: GraphSAGE实现 (4-6小时)

**核心任务**
```bash
[ ] 创建 src/models/sage.py
[ ] 实现 GraphSAGE 类
    [ ] 使用 SAGEConv 层
    [ ] 邻居采样策略
    [ ] 不同聚合器（mean/max/lstm）
[ ] 创建 train_sage.py
[ ] 训练GraphSAGE模型
[ ] 性能评估
```

**检查清单**
- [ ] `src/models/sage.py` 实现完成
- [ ] 支持多种聚合器
- [ ] `train_sage.py` 运行成功
- [ ] 测试集 AUC ≥ 0.915
- [ ] 可扩展性验证

---

### Week 3 总检查

```
Week 3 完成度检查
=================

Day 1-3: GCN
[ ] 同构图构建 ✅
[ ] GCN模型实现 ✅  
[ ] GCN训练完成 ✅
[ ] AUC ≥ 0.910 ✅

Day 4-5: GAT
[ ] GAT模型实现
[ ] GAT训练完成
[ ] AUC ≥ 0.920
[ ] 注意力分析

Day 6-7: GraphSAGE
[ ] SAGE模型实现
[ ] SAGE训练完成
[ ] AUC ≥ 0.915

性能对比表
模型          AUC      F1       训练时间
MLP (基线)   0.909    0.516    5分钟
GCN          ____     ____     ____
GAT          ____     ____     ____
GraphSAGE    ____     ____     ____

如果Week 3全部✅，进入Week 4！
```

---

## 🎯 Week 4: 异构GNN + 多任务学习

### Day 8-10: HAN实现 (6-8小时)

**核心任务**
```bash
[ ] 创建 src/graph/heterogeneous_graph_builder.py
[ ] 构建完整异构图（5种节点，5种边）
[ ] 创建 src/models/han.py  
[ ] 实现 HAN 类
    [ ] 节点级注意力
    [ ] 语义级注意力
    [ ] 元路径定义
[ ] 创建 train_han.py
[ ] 训练HAN模型
```

**检查清单**
- [ ] 异构图构建完成
- [ ] HAN模型实现完成
- [ ] 元路径定义合理
- [ ] 训练成功
- [ ] 测试集 AUC ≥ 0.930

**性能目标**
```
HAN性能目标
===========
测试集 AUC:       ≥ 0.930   [实际: _____ ]
测试集 F1:        ≥ 0.570   [实际: _____ ]
相比GAT AUC提升:  ≥ +1%     [实际: _____ ]
```

---

### Day 11-12: 多任务学习 (4-6小时)

**核心任务**
```bash
[ ] 实现多任务损失函数
    [ ] 任务1: 离职预测 (主任务)
    [ ] 任务2: 岗位偏好预测 (辅助任务)
[ ] 创建 train_multitask.py
[ ] 联合训练优化
[ ] 消融实验
```

**检查清单**
- [ ] 多任务损失实现完成
- [ ] 权重平衡合理
- [ ] 联合训练成功
- [ ] 消融实验完成
- [ ] 多任务相比单任务有提升

---

### Day 13-14: 模型对比与最终报告 (4-6小时)

**核心任务**
```bash
[ ] 完整性能对比表
[ ] 消融实验分析
[ ] 可视化对比图表
[ ] 生成最终报告
[ ] 代码整理与文档
```

**最终检查清单**
- [ ] 所有模型性能表格完成
- [ ] 消融实验结果分析
- [ ] 可视化图表生成
- [ ] 最终报告 `Week_3-4_Summary.md` 完成
- [ ] 所有代码有详细注释
- [ ] README更新

---

## 📊 最终性能对比表

```
完整模型性能对比
===============================================
模型          AUC      F1       Precision  Recall    训练时间
MLP (基线)   0.909    0.516    0.400      0.727     5min
GCN          ____     ____     ____       ____      ____
GAT          ____     ____     ____       ____      ____
GraphSAGE    ____     ____     ____       ____      ____
HAN          ____     ____     ____       ____      ____
HAN+多任务   ____     ____     ____       ____      ____

目标:
- 最佳AUC ≥ 0.930
- 相比MLP提升 ≥ 2.3%
- 至少一个模型F1 ≥ 0.570
```

---

## 🎓 Week 3-4 完成标准

### 必须完成 (Must Have)

- [ ] ✅ GCN实现并训练（AUC ≥ 0.910）
- [ ] ✅ GAT实现并训练（AUC ≥ 0.920）
- [ ] ✅ HAN实现并训练（AUC ≥ 0.930）
- [ ] ✅ 所有模型评估报告
- [ ] ✅ 性能对比表格
- [ ] ✅ 代码完整可运行

### 应该完成 (Should Have)

- [ ] GraphSAGE实现
- [ ] 多任务学习
- [ ] 消融实验
- [ ] 可视化分析

### 可选完成 (Nice to Have)

- [ ] 超参数优化
- [ ] 可解释性分析
- [ ] 注意力可视化
- [ ] 特征重要性分析

---

## 🚀 快速启动命令

```bash
# Week 3 Day 1-3
python src/graph/homogeneous_graph_builder.py
python src/models/gcn.py
python train_gcn.py

# Week 3 Day 4-5
python src/models/gat.py
python train_gat.py

# Week 3 Day 6-7
python src/models/sage.py
python train_sage.py

# Week 4 Day 8-10
python src/graph/heterogeneous_graph_builder.py
python src/models/han.py
python train_han.py

# Week 4 Day 11-12
python train_multitask.py

# Week 4 Day 13-14
python generate_final_report.py
```

---

## 📚 参考资料

### 论文
- **GCN**: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
- **GAT**: Graph Attention Networks (ICLR 2018)
- **GraphSAGE**: Inductive Representation Learning on Large Graphs (NeurIPS 2017)
- **HAN**: Heterogeneous Graph Attention Network (WWW 2019)

### 代码示例
- PyTorch Geometric Examples: https://github.com/pyg-team/pytorch_geometric/tree/master/examples
- HAN官方实现: https://github.com/Jhy1993/HAN

---

## 🎉 完成Week 3-4后你将获得

### 技能提升
✅ **GNN原理** - 深入理解图神经网络  
✅ **PyG实践** - 熟练使用PyTorch Geometric  
✅ **异构图** - 掌握异构图建模  
✅ **多任务学习** - 实现联合优化

### 代码资产
✅ **3-4个GNN模型** - 可复用的模型库  
✅ **完整训练流程** - 标准化训练代码  
✅ **评估工具** - 通用评估框架  
✅ **可视化工具** - 丰富的图表

### 性能提升
✅ **AUC提升** - 从0.909到0.93+  
✅ **F1提升** - 从0.516到0.57+  
✅ **Precision提升** - 从0.40到0.50+

---

**准备好开始Week 3-4了吗？** 🚀

立即开始Day 1: [Week3_4_Implementation_Guide.md](./Week3_4_Implementation_Guide.md)

---

*最后更新: 2025-10-18*  
*版本: v1.0*  
*状态: ✅ 就绪*
