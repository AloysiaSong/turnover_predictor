# Week 5-6 Implementation Plan (离职预测 + 岗位偏好异构GNN)

## 总览
目标：把现有的同构 GCN 升级为“员工-岗位-公司属性”的异构图，并构建联合离职预测 + 岗位偏好排序的多任务 GNN，提升整体表现与可解释性。

时间窗口：Week 5-6，共 10 个工作日（可根据节奏调优）。

---

## 阶段拆解

### 1. 数据管线升级（Day 1-2）
- [ ] **图结构重建**
  - 扩展 `EdgeBuilder` 或新增 `hetero_graph_builder.py`：
    - 节点类型：`employee`、`current_job`、`post_type (14)`、`company_size`、`company_type`（可扩展其它属性）。
    - 边：
      - `employee → current_job`（现岗位，没有可填 NULL 时映射到 `current_job_unknown`）
      - `employee → post_type`（14 岗位类别，多选 + 偏好）
      - `post_type → company_size/company_type`（岗位-公司属性，或使用 `current_job` 中携带的属性边）
      - 偏好 pairwise 数据：基于 `preference_ranks` 构造 `(employee, prefer_post, disprefer_post)` 三元组，保存成 `preference_triples.pt` 或 CSV。
- [ ] **特征准备**
  - 员工 47 维特征：写新脚本 `scripts/standardize_features.py`（或在 `feature_pipeline` 内改造）完成 `StandardScaler` 拟合与存储 `scaler.pkl`。
  - 额外派生特征（可选）：技能熟练度均值、最大-最小差、经济损失得分均值、偏好分数统计等。
  - 岗位/公司属性可采用 One-hot 或 learnable embedding，由模型处理。

### 2. 模型开发（Day 3-6）
- [ ] **模型结构**
  - 新建 `src/models/hetero_gnn.py`：
    - 使用 PyG `HeteroConv` / `HAN` / `HGT`（先选 HGT，后续可替换）。
    - 为不同关系定义专用 GNN 层；最后返回 `employee_emb`、`current_job_emb`、`post_type_emb`。
  - 辅助模块：
    - `src/models/layers/feature_normalizer.py`：包装 scaler，训练时加载 `scaler.pkl`。
    - Job/Company embedding 初始化逻辑。
- [ ] **多任务头**
  - 离职头：`MLP(concat(employee_emb, current_job_emb)) → sigmoid`。
  - 偏好头：
    - Pairwise margin ranking：对 `(prefer, disprefer)`，计算 `score = f(employee_emb, post_emb)`，目标 `score_pref > score_dispref`。
    - 可选边分类：`score(edge)`，用 BCE 判断 prefer/disprefer。
  - 辅助头（可选）：预测 `preference_ranks` 的排序（Reg/Classification）。
- [ ] **Loss & Scheduler**
  - Loss 合成：`total = α * loss_turnover + β * loss_preference (+ γ * loss_aux)`，α/β/γ 可配置。
  - 支持 BCEWithLogits + Focal；Margin Ranking Loss / BPR；辅助 loss 使用 MSE 或 CE。
  - Scheduler 沿用 ReduceLROnPlateau 或引入 CosineAnnealing。

### 3. 训练脚本 & 配置（Day 7-8）
- [ ] `scripts/train_gnn_v5.py`
  - 解析超参：模型类型、各损失权重、正则、学习率、scheduler。
  - 加载 `hetero_graph.pt`，应用特征归一化。
  - 生成 pairwise mini-batch（pref/dispref），可通过 `torch.utils.data.Dataset` 实现。
  - 日志：离职 loss/F1、偏好 loss/AUC（或 top-K precision）、辅助指标；保存 best model & history。
  - 阈值优化：训练后扫描验证集，保存最优阈值。
- [ ] 配置管理
  - 建议添加 YAML/JSON 配置（`configs/hetero/default.yaml`），便于批量实验。

### 4. 评估与可视化（Day 9）
- [ ] 更新 `week12_reporter.py` 或新增脚本：
  - 报告离职指标：`accuracy, precision, recall, F1, AUROC, AUPR`。
  - 报告偏好指标：`pairwise accuracy, NDCG@K, recall@K`。
  - 绘制 emb 可视化（可使用 t-SNE/Umap）。
- [ ] 对比基线：
  - 与 v2/v4 的同构模型对比验证集指标，记录改进幅度。

### 5. 文档 & 交付准备（Day 10）
- [ ] 更新 README / `reports/week5_6_plan.md` → 本文件。
- [ ] 编写 `docs/hetero_gnn_usage.md`：训练/评估命令、数据准备说明。
- [ ] 整理待办 & 风险：
  - Graph 构建失败 / 缺失特征的兜底处理。
  - 模型爆显存：可通过 mini-batch、采样或更轻模型解决。

---

## 关键实现建议

1. **图保存格式**  
   - 使用 `torch.save({'graph': data, 'preference_pairs': triples, 'meta': {...}}, path)` 方式，便于一次性加载所有信息。
2. **pairwise 数据加载**  
   - 自定义 `PreferencePairDataset`：`__getitem__` 返回 `(employee_id, prefer_post_id, disprefer_post_id)`。
   - 训练时随机采样或遍历 7 × 500 的偏好对。
3. **可配置化**  
   - 在模型文件内用 dataclass 或 config dict 构建，保持脚本简洁。
4. **多任务权重**  
   - 初始可设 `α=1.0, β=0.5`，后续根据验证集表现调节；如偏好 loss 太小，可提升 β。
5. **Feature Normalization**  
   - 训练集拟合 scaler，保存到 `data/processed/feature_scaler.pkl`；脚本加载后对所有节点特征正态化。

---

## 预期产出

- `data/processed/hetero_graph.pt`
- `data/processed/feature_scaler.pkl`
- `src/models/hetero_gnn.py` + 相应 trainer
- `scripts/train_gnn_v5.py`
- 配置文件、评估报表、更新文档

如有剩余时间，可拓展：加入公司属性节点、岗位说明文本 embedding、或尝试 GAT/HGT 多模型对比。
