# GNN模型可解释性指南

**版本**: 1.0
**日期**: 2025-10-19
**目的**: 解释"为什么该员工被判为高风险"

---

## 📋 目录

1. [概述](#概述)
2. [解释方法](#解释方法)
3. [使用指南](#使用指南)
4. [示例分析](#示例分析)
5. [技术细节](#技术细节)
6. [未来扩展](#未来扩展)

---

## 概述

### 为什么需要可解释性?

在员工离职预测场景中,仅提供"该员工离职风险高"是不够的,我们需要回答:

1. **为什么**这个员工被判为高风险?
2. **哪些特征**对预测贡献最大?
3. **哪些关系**(图结构)影响了预测?
4. **偏好推荐**的理由是什么?

### 我们的解释框架

```
┌─────────────────────────────────────────────────────────────┐
│                  GNN可解释性框架                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 特征级贡献 (Feature-Level)                               │
│     → 哪些员工特征增加/降低离职风险?                          │
│     → 线性权重分解                                            │
│                                                               │
│  2. 图结构贡献 (Neighbor-Level)                              │
│     → 哪些岗位/公司关系影响预测?                              │
│     → 注意力权重分析                                          │
│                                                               │
│  3. 偏好解释 (Preference)                                    │
│     → 为什么员工偏好岗位A胜过岗位B?                           │
│     → Pairwise分数对比                                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 解释方法

### 1. 特征级贡献分析

**方法**: 线性权重分解

**原理**:
```python
# TurnoverHead的第一层是Linear
logit = W · concat([employee_emb, job_emb]) + b

# 每个特征的贡献
contribution_i = W_i * x_i
```

**输出**:
- Top正向特征 (增加离职风险)
- Top负向特征 (降低离职风险)
- Bias贡献

**示例**:
```json
{
  "employee_id": 0,
  "turnover_probability": 0.19,
  "prediction": "Low Risk",
  "top_positive_features": [
    {"name": "skill_freq_15", "contribution": 0.056},
    {"name": "skill_freq_4", "contribution": 0.053}
  ],
  "top_negative_features": [
    {"name": "skill_prof_1", "contribution": -0.085},
    {"name": "feature_248", "contribution": -0.077}
  ]
}
```

**解读**:
- 负向贡献占优 → 预测为低风险
- `skill_prof_1` (-0.085) 是最强保护因素

---

### 2. 图结构贡献分析

**方法**: 邻居重要性分析

**原理**:
```python
# 使用embedding相似度作为重要性代理
importance = cosine_similarity(employee_emb, neighbor_emb)
```

**输出**:
- 当前岗位的重要性分数
- 关键公司关系
- 偏好岗位类型

**示例**:
```json
{
  "employee_id": 0,
  "important_relations": [
    {
      "relation_type": "assigned_to_current_job",
      "target_id": 42,
      "importance_score": 0.73
    }
  ]
}
```

**解读**:
- 重要性分数0.73表示当前岗位与员工匹配度较高
- 高匹配度 → 降低离职风险

---

### 3. 偏好解释

**方法**: Pairwise分数对比

**原理**:
```python
# Dual-Head模型使用dot product
score_A = employee_emb · post_A_emb
score_B = employee_emb · post_B_emb

margin = score_A - score_B  # 偏好强度
```

**输出**:
- 偏好岗位vs非偏好岗位
- 分数差距(margin)
- 对齐维度分析

**示例**:
```json
{
  "employee_id": 15,
  "preferred_post": 23,
  "dispreferred_post": 67,
  "preference_score": 0.82,
  "dispreference_score": 0.31,
  "margin": 0.51,
  "confidence": "High"
}
```

**解读**:
- Margin 0.51 > 0.5 → 高置信度偏好
- 员工与岗位23的embedding高度对齐

---

## 使用指南

### 快速开始

**步骤1: 训练模型** (如已训练可跳过)
```bash
python scripts/train_dual_head.py
```

**步骤2: 生成解释**
```bash
python scripts/explain_predictions.py \
    --run-dir outputs/dual_head/dual_head_main \
    --explain-ids 0 5 10 15 20 \
    --visualize
```

**步骤3: 查看结果**
```bash
ls outputs/dual_head/dual_head_main/explanations/
```

### 输出文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `employee_XXXX_features.json` | 特征贡献详情 | 了解离职风险来源 |
| `employee_XXXX_neighbors.json` | 邻居重要性 | 图结构影响分析 |
| `employee_XXXX_importance.png` | 特征可视化 | 直观展示 |
| `employee_XXXX_subgraph.png` | 关系子图 | 图结构可视化 |
| `preference_explanations.json` | 偏好解释 | 理解推荐理由 |
| `preference_comparison_X.png` | 偏好对比图 | 可视化偏好 |
| `summary.json` | 总结报告 | 高层次概览 |

---

## 示例分析

### 案例1: 低风险员工 (Employee 0)

**预测**: Low Risk (p=0.19)

**关键发现**:

**Top保护因素** (降低离职风险):
```
1. skill_prof_1:    -0.085  (高技能熟练度)
2. feature_248:     -0.077  (未知特征)
3. feature_220:     -0.053
```

**Top风险因素** (增加离职风险):
```
1. feature_153:     +0.064
2. skill_freq_15:   +0.056  (技能使用频率)
3. skill_freq_4:    +0.053
```

**图结构贡献**:
```
- 当前岗位匹配度: 0.73 (高)
→ 岗位满意度高,降低离职风险
```

**业务解读**:
- ✅ 该员工技能熟练度高 (`skill_prof_1`)
- ✅ 当前岗位匹配度好 (0.73)
- ⚠️ 某些技能使用频率可能偏高,需要关注工作强度
- **结论**: 整体低风险,但可优化工作量分配

---

### 案例2: 偏好推荐解释 (Employee 15)

**偏好**: 岗位23 > 岗位67

**分数**:
```
岗位23 (偏好):      0.82
岗位67 (非偏好):    0.31
Margin:             0.51 (高置信度)
```

**embedding对齐分析**:
```
Top对齐维度: [45, 78, 102, 23, 67]
→ 这些维度上,员工与岗位23高度相似
```

**业务解读**:
- ✅ 员工与岗位23的技能/兴趣高度匹配
- ⚠️ 岗位67匹配度较低
- **推荐**: 优先考虑岗位23类型的机会

---

## 技术细节

### 实现架构

```python
# 核心模块
src/models/
├── explanations.py          # 核心解释逻辑
│   ├── FeatureContributionAnalyzer
│   ├── AttentionWeightExtractor
│   └── PreferenceExplainer
│
├── visualization.py         # 可视化工具
│   ├── plot_explanation_subgraph()
│   └── plot_preference_comparison()
│
└── scripts/
    └── explain_predictions.py  # 命令行接口
```

### 关键函数

**1. 特征贡献计算**
```python
def compute_contributions(
    employee_embeddings: torch.Tensor,
    job_embeddings: torch.Tensor,
    employee_ids: List[int],
) -> List[Dict]:
    # 获取线性层权重
    weights = turnover_head.net[0].weight

    # 计算贡献: w * x
    contributions = weights * concat_embedding

    # 排序并返回top-k
    return top_k_features
```

**2. 邻居重要性**
```python
def analyze_neighbor_importance(
    data: HeteroData,
    embeddings: Dict[str, Tensor],
    employee_ids: List[int],
) -> List[Dict]:
    # 计算cosine相似度
    similarity = cosine_similarity(
        employee_emb,
        neighbor_emb
    )

    return neighbor_importance
```

**3. 偏好解释**
```python
def explain_pairwise_preference(
    employee_embeddings: Tensor,
    post_embeddings: Tensor,
    triples: Tensor,
) -> List[Dict]:
    # 计算pairwise分数
    pref_score = employee_emb · post_pref_emb
    disp_score = employee_emb · post_disp_emb
    margin = pref_score - disp_score

    return explanations
```

---

## 可视化示例

### 1. 特征重要性图

![Feature Importance](examples/employee_0000_importance.png)

**说明**:
- 绿色条: 增加离职风险的特征
- 红色条: 降低离职风险的特征
- 长度: 贡献大小

### 2. 关系子图

![Subgraph](examples/employee_0000_subgraph.png)

**说明**:
- 红色节点: 高风险员工
- 绿色节点: 低风险员工
- 蓝色节点: 当前岗位
- 边宽度: 关系重要性

### 3. 偏好对比图

![Preference](examples/preference_comparison_0.png)

**说明**:
- 绿色柱: 偏好岗位分数
- 红色柱: 非偏好岗位分数
- Margin: 分数差距

---

## 未来扩展 (TODO)

### 1. 全局解释方法

**SHAP (SHapley Additive exPlanations)**
```python
# TODO: 集成SHAP库
import shap

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(employee_features)
```

**优势**:
- 理论保证 (Shapley值)
- 全局一致性
- 特征交互分析

---

### 2. Permutation Importance

**方法**: 打乱特征观察性能下降
```python
# TODO
def permutation_importance(model, data, feature_idx):
    baseline_score = evaluate(model, data)

    # Permute feature
    data_permuted = permute_feature(data, feature_idx)
    permuted_score = evaluate(model, data_permuted)

    importance = baseline_score - permuted_score
    return importance
```

---

### 3. 注意力可视化

**要求**: 修改HGTConv保存attention权重
```python
# TODO: 修改src/models/hetero_gnn.py
class HGTConv(nn.Module):
    def forward(self, x_dict, edge_index_dict):
        # ... existing code ...

        # Save attention weights
        self.last_attention = attention_weights

        return h_dict
```

**可视化**:
```python
# 提取并可视化attention
attention = model.gnn_layers[-1].last_attention
plot_attention_heatmap(attention)
```

---

### 4. Integrated Gradients

**方法**: 基于梯度的归因
```python
# TODO: 使用Captum库
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(
    inputs=employee_embedding,
    target=turnover_prediction
)
```

---

### 5. GraphMask

**方法**: 学习重要子图
```python
# TODO: 实现GraphMask
class GraphMask(nn.Module):
    def forward(self, graph):
        # Learn binary mask for edges/nodes
        mask = learn_mask(graph)

        # Prune graph
        important_subgraph = apply_mask(graph, mask)

        return important_subgraph
```

---

## 与Baseline对比

### MLP/XGBoost的解释

**特征重要性**:
```python
# XGBoost
feature_importance = model.feature_importances_

# MLP (需要额外工具)
# 1. Permutation Importance
# 2. LIME
# 3. SHAP
```

**限制**:
- ❌ 无法利用图结构信息
- ❌ 无法解释关系重要性
- ❌ 无法捕捉higher-order patterns

### GNN的额外洞察

**优势**:
- ✅ **图结构贡献**: 哪些岗位/公司关系重要
- ✅ **邻居影响**: 同事/团队对离职的影响
- ✅ **多跳传播**: 间接关系的作用
- ✅ **偏好解释**: embedding空间的语义解释

---

## API参考

### 命令行接口

```bash
python scripts/explain_predictions.py \
    --run-dir <path>              # 模型目录
    --explain-ids <id1> <id2>     # 员工ID列表
    --output-dir <path>           # 输出目录(可选)
    --top-k <n>                   # Top特征数(默认10)
    --visualize                   # 生成可视化
```

### Python API

```python
from src.models.explanations import (
    FeatureContributionAnalyzer,
    AttentionWeightExtractor,
    PreferenceExplainer,
    generate_explanation_report,
)

# 生成完整报告
generate_explanation_report(
    model=trained_model,
    turnover_head=turnover_head,
    preference_head=preference_head,
    data=graph_data,
    employee_ids=[0, 5, 10],
    triples=preference_triples,
    scaler_path="data/processed/feature_scaler.pkl",
    feature_names_path="data/processed/feature_names.txt",
    save_dir="outputs/explanations",
)
```

---

## 常见问题

### Q1: 解释是否可靠?

**A**: 我们的解释基于:
1. **线性近似**: TurnoverHead第一层的线性权重
2. **embedding相似度**: 作为注意力的代理
3. **Dot product分解**: Preference头的直接计算

这些方法在实践中被广泛验证,但仍是**近似解释**,不是精确因果关系。

### Q2: 如何处理高维embedding?

**A**:
- 我们关注**top-k特征** (默认10)
- 使用**绝对贡献值**排序
- 提供**可视化**辅助理解

### Q3: 负向贡献意味着什么?

**A**:
- **正向贡献**: 增加离职风险
- **负向贡献**: 降低离职风险(保护因素)
- **Bias**: 模型的基线倾向

### Q4: 如何验证解释的正确性?

**A**:
1. **业务验证**: 与HR专家讨论
2. **对比实验**: 修改关键特征观察预测变化
3. **A/B测试**: 在真实场景中测试

---

## 参考文献

1. **GNN Explainability**:
   - GNNExplainer (Ying et al., NeurIPS 2019)
   - GraphMask (Schlichtkrull et al., ICML 2021)

2. **General XAI**:
   - SHAP (Lundberg & Lee, NIPS 2017)
   - LIME (Ribeiro et al., KDD 2016)
   - Integrated Gradients (Sundararajan et al., ICML 2017)

3. **Attention Visualization**:
   - Attention Is All You Need (Vaswani et al., 2017)
   - Analyzing and Interpreting Neural Networks for NLP (Belinkov & Glass, 2019)

---

## 总结

我们实现了一套**完整的GNN可解释性框架**:

✅ **特征级**: 线性权重分解
✅ **图结构级**: 邻居重要性分析
✅ **偏好级**: Pairwise对比解释
✅ **可视化**: 自动生成图表
✅ **易用性**: 简单命令行接口

**下一步**:
1. 实施TODO中的高级方法 (SHAP, Attention等)
2. 与业务团队合作验证解释
3. 构建交互式dashboard

---

*文档版本: 1.0*
*最后更新: 2025-10-19*
*维护者: GNN Explainability Team*
