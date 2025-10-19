# HGNN 离职预测与岗位偏好项目

> 更新时间：2025-10-19 15:41:20 CEST

本项目围绕 500 名员工的调查数据，构建异构图神经网络（HGNN）以同时完成离职预测与岗位偏好建模。仓库包含从原始数据清洗、图构建、特征工程到模型训练与评估的一整套流水线，并支持多种实验脚本（如阈值优化、集成、对比实验等）。

---

## 🚀 快速上手

1. **创建环境**
   ```bash
   conda create -n hgnn_project python=3.10
   conda activate hgnn_project
   pip install -r requirements.txt
   ```
2. **生成图数据（同构图）**
   ```bash
   python scripts/train_gcn.py --data-path data/processed/homo_graph.pt
   ```
3. **运行新版 GCN 训练**
   ```bash
   python scripts/train_gcn_v2.py \
     --data-path data/processed/homo_graph.pt \
     --save-dir outputs/models \
     --eval-dir outputs/evaluation
   ```
4. **（可选）生成带岗位偏好的多任务图**
   ```bash
   python scripts/prepare_multitask_data.py \
     --mode real \
     --original-csv data/raw/originaldata.csv \
     --graph-path data/processed/homo_graph.pt \
     --output-path data/processed/homo_graph_with_preferences.pt
   ```

---

## 📦 目录索引

| 目录 | 说明 |
|------|------|
| `configs/` | 训练与特征工程的 YAML 配置 |
| `data/` | 原始、处理中及图数据文件 |
| `outputs/` | 训练好的模型、日志与评估结果 |
| `scripts/` | 命令行脚本（训练、评估、数据准备等） |
| `src/` | 项目核心代码：模型、数据处理、特征工程 |
| `reports/` | 阶段性分析与可行性报告 |

---

## 🧱 数据与特征流水线

- `src/data_processing/load_data.py`：加载原始 CSV 并生成 PyTorch Geometric `Data` 对象。
- `src/data_processing/edge_builder.py`：构建员工、岗位、公司等节点之间的边。
- `src/data_processing/create_masks.py`：生成 `train/val/test` 掩码，默认 340/60/100 划分。
- `src/data_processing/data_splitter.py`：支持多种随机切分策略。
- `src/features/feature_extractor.py` & `src/data/feature_pipeline.py`：55 维员工特征提取与可选的特征增强（离散化、交互项）。
- `scripts/prepare_multitask_data.py`：为多任务场景添加 7 个岗位偏好排序；提供真实提取与模拟数据两种模式。

---

## 🧠 模型与训练

| 文件 | 内容摘要 |
|------|----------|
| `src/models/gcn.py` | 支持多层 GCN、Dropout、Edge Dropout 与 Feature Dropout 等配置。 |
| `src/models/trainer2.py` | GCN 专用训练器，含 ReduceLROnPlateau 调度、早停、`predict` 助手等。 |
| `src/models/trainer.py` | MLP 训练器，同时提供 `compute_metrics` 用于二分类评估。 |
| `src/models/threshold_optimizer.py` | 扫描阈值获取最佳 F1 / Recall / Precision。 |
| `src/models/week12_reporter.py` | 训练历史与性能指标的报告生成工具。 |

核心训练脚本：

- `scripts/train_gcn.py`：最初版本的同构图 GCN 训练流程。
- `scripts/train_gcn_v2.py`：增强版训练，包含配置打印、早停、阈值扫描与完整评估存档。
- `scripts/train_gcn_v3.py`：预留的多任务训练入口（结合偏好数据）。
- `scripts/train_mlp_baseline.py`、`train_focal_loss.py`、`train_smote.py`、`train_ensemble.py`：对比/增强实验。

---

## 📊 评估与分析

- `src/models/evaluator.py` / `evaluator2.py`：生成 ROC、PR 曲线、混淆矩阵与分类报告。
- `scripts/threshold_optimization.py`：自动搜索验证集最优阈值并输出 JSON。
- `outputs/models/<run_id>/`：保存最佳模型权重 (`best_model.pt`) 与训练曲线 (`training_history.json`)。
- `outputs/evaluation/<run_id>/results.json`：记录阈值扫描、验证 F1 与测试集指标。

---

## 📝 报告与文档

- `FINAL_HGNN_Feasibility_Report.md`：针对 500 样本完整数据的可行性评估（强烈推荐阅读）。
- `Week1_2_Implementation_Guide*.md`、`Week3_4_Implementation_Guide.md`：按周拆分的实施手册。
- `PROJECT_DELIVERY_CHECKLIST.md`：交付前自查清单。

---

## 🔄 常见工作流

1. **数据准备**
   - 使用 `load_data.py` 清洗并缓存图数据。
   - 如需多任务，在 `prepare_multitask_data.py` 生成带 `preference_ranks` 的图。
2. **模型训练**
   - 选择合适的配置并运行 `train_gcn_v2.py`（支持命令行覆盖参数）。
   - 训练过程中自动保存最佳模型与学习率调度日志。
3. **评估与分析**
   - 使用 `threshold_optimizer.py`、`week12_reporter.py` 生成报告。
   - 通过 `evaluator2.py` 导出混淆矩阵与 ROC/PR 曲线图。
4. **实验管理**
   - `configs/` 下可维护多套 YAML 配置。
   - 将结果记录在 `outputs/evaluation/`，方便复现与比较。

---

## ✅ 最新改动摘要（截至 2025-10-19）

- `trainer2.GCNTrainer.predict` 支持直接获取任意 mask 上的概率与标签，便于阈值搜索与统计分析。
- `trainer.compute_metrics` 提供统一的分类指标，供脚本复用。
- `prepare_multitask_data.py` 修复提取逻辑缩进问题，能更稳健地识别 `S1..S7` 样式列或自动回退到模拟数据。

---

## 🤝 贡献与扩展

- 欢迎在 `src/models` 中添加新的图模型（如 GAT、GraphSAGE）。
- 可在 `scripts/` 下创建实验脚本，并复用 `trainer2.GCNTrainer` 的训练循环。
- 建议在新增数据处理步骤后，更新 `docs/` 或 `reports/` 中的指南，保持交付材料同步。

---

如有问题或需要进一步自动化（数据版本管理、实验记录、可视化等），可在 `PROJECT_DELIVERY_CHECKLIST.md` 中新增任务并提交 Issue/PR。祝实验顺利！
