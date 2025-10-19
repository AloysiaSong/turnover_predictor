# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Heterogeneous Graph Neural Network (HGNN) project for employee turnover prediction and job preference analysis using PyTorch Geometric. The project uses a Chinese employee dataset (originaldata.csv, 500 samples) with 55 features to predict employee turnover and analyze job preferences across 7 scenario-based tasks.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n hgnn python=3.9
conda activate hgnn

# Install dependencies
pip install -r requirements.txt
```

### Data Processing Pipeline
```bash
# Step 1: Load and extract features
python src/data_processing/load_data.py

# Step 2: Extract features (creates employee_features.npy)
python src/features/feature_extractor.py

# Step 3: Extract labels (creates y_turnover_binary.npy)
python src/data_processing/label_extractor.py

# Step 4: Build edges (creates edge files in data/processed/edges/)
python src/data_processing/edge_builder.py

# Step 5: Create train/val/test splits (340/60/100)
python src/data_processing/data_splitter.py

# Step 6: Build homogeneous graph (creates homo_graph.pt)
python src/graph/homogeneous_graph_builder.py
```

### Training Models

```bash
# Train MLP baseline (Week 1-2)
python src/models/train_mlp_baseline.py

# Train GCN model (Week 3-4)
python scripts/train_gcn.py

# Train GCN with advanced features
python scripts/train_gcn_v2.py

# Train with focal loss (handles class imbalance)
python scripts/train_focal_loss.py

# Train with SMOTE (synthetic oversampling)
python scripts/train_smote.py

# Train ensemble models
python scripts/train_ensemble.py
```

### Model Optimization

```bash
# Compare different architectures
python scripts/compare_architectures.py

# Optimize classification threshold
python scripts/threshold_optimization.py
```

### Testing Single Components
```bash
# Test data loading
python src/data_processing/load_data.py

# Test feature extraction
python src/features/feature_extractor.py

# Test model creation
python src/models/mlp_baseline.py
python src/models/gcn.py
```

## Architecture Overview

### Data Pipeline Architecture

The project follows a **multi-stage feature extraction and graph construction** approach:

1. **Raw Data** → CSV file with Chinese encoding (GBK), requiring `skiprows=1` due to duplicate header row
2. **Feature Extraction** → 55-dimensional employee features from questionnaire responses (Q1-Q17, Q5 multi-hot encoding)
3. **Label Extraction** → Binary turnover labels from Q30 ("会"=1 turnover, "不会"=0 stay), highly imbalanced (56:444 ratio ~7.9:1)
4. **Edge Construction** → Multiple edge types connecting employees to job types, company sizes, and company types
5. **Graph Building** → Homogeneous graph using hybrid edge strategy (attribute-based + k-NN + similarity-based)

### Model Architecture Progression

The project implements a **progressive learning strategy** from simple to complex:

**Week 1-2: MLP Baseline**
- Architecture: 3-layer MLP `[input → 128 → 64 → 32 → 1]`
- Handles class imbalance with `pos_weight=7.9` in BCE loss
- Expected performance: AUC 0.72-0.78, F1 0.35-0.45
- Located in: `src/models/mlp_baseline.py`

**Week 3-4: Homogeneous GCN**
- Uses same-type nodes (all employees) with similarity-based edges
- Implements DropEdge for regularization
- Hybrid edge strategy combining: attribute edges (weight 1.5) + k-NN edges + high-similarity edges
- Architecture variants: shallow (1-layer), default (2-layer), deep (3-layer), very_deep (4-layer)
- Expected performance: AUC 0.76-0.83, F1 0.40-0.50
- Located in: `src/models/gcn.py`, `src/graph/homogeneous_graph_builder.py`

**Future: Heterogeneous GNN++**
- Multi-node types: Employee, PostType, CompanySize, CompanyType, HypotheticalPost
- Multi-task learning: turnover classification + job preference ranking
- Expected performance: AUC 0.82-0.86, F1 0.48-0.55

### Key Design Patterns

**Data Loading Pattern:**
- All data loaders use hardcoded absolute paths (e.g., `/Users/yu/code/code2510/gnn/data/...`)
- When modifying data loading code, update these paths or make them relative/configurable
- CSV encoding is always `gbk` with `skiprows=1`

**Training Pattern:**
- All trainers use early stopping (default patience=15-20 epochs)
- Models are saved to `models/{model_name}/` or `outputs/models/`
- Evaluation results are saved to `results/{model_name}/` or `outputs/evaluation/`
- Training history is always saved as JSON for later analysis

**Evaluation Pattern:**
- Multiple metrics tracked: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- Separate evaluators for MLP (`evaluator.py`) and GCN (`evaluator2.py`)
- Visualization includes: confusion matrix, ROC curve, PR curve, training curves

**Graph Construction Pattern:**
- Homogeneous graphs use hybrid edge strategy by default
- Edge weights are normalized and can be combined from multiple sources
- Graph validation always checks: connectivity, isolated nodes, degree distribution, label balance

## Important Implementation Details

### Class Imbalance Handling

The dataset has severe class imbalance (7.9:1). All models must address this:
- Use `pos_weight=7.9` in BCEWithLogitsLoss
- Optionally use Focal Loss (alpha=0.25, gamma=2) via `scripts/train_focal_loss.py`
- Optionally use SMOTE for synthetic oversampling via `scripts/train_smote.py`
- Never evaluate using accuracy alone; prioritize AUC, F1, and precision/recall

### Feature Dimensions

- **Input features**: 47 or 55 dimensions depending on extraction method (check feature_extractor.py)
- **MLP baseline** uses 47 features
- **GCN models** typically use the same features as node features
- Features are always standardized using `StandardScaler`

### Data Splits

- **Train**: 340 samples (68%)
- **Val**: 60 samples (12%)
- **Test**: 100 samples (20%)
- Splits use stratification to maintain label balance
- Stored as both `.npy` (indices) and `.pt` (boolean masks) formats

### Graph Edge Strategies

When building homogeneous graphs (`src/graph/homogeneous_graph_builder.py`):

1. **Attribute edges**: Connect employees with same job type (weight 1.0), company size (0.7), or company type (0.7)
2. **k-NN edges**: Connect each employee to k nearest neighbors (default k=10) based on feature similarity
3. **Similarity edges**: Connect employees with cosine similarity above threshold (default 0.6)
4. **Hybrid strategy** (recommended): Combines all three, with attribute edges weighted 1.5x

### Multi-Task Learning (Future)

The dataset supports:
1. **Primary task**: Binary turnover classification (Q30)
2. **Secondary task**: Job preference ranking from 7 scenario-based tasks (Q18-Q29), providing 3,500 preference pairs
3. **Optional task**: Turnover probability regression (Q31_1)

## File Organization

```
data/
├── raw/originaldata.csv              # Source data (GBK encoding)
├── processed/
│   ├── employee_features.npy         # 55-dim features
│   ├── y_turnover_binary.npy         # Binary labels
│   ├── homo_graph.pt                 # Homogeneous graph
│   └── edges/                        # Edge data
└── splits/
    ├── train_idx.npy, val_idx.npy, test_idx.npy  # Split indices
    └── train_mask.pt, val_mask.pt, test_mask.pt  # Boolean masks

src/
├── data_processing/                  # Data loading and preprocessing
├── features/                         # Feature extraction
├── models/                           # Model definitions and trainers
└── graph/                            # Graph construction

scripts/                              # Training scripts for experiments
outputs/                              # Model checkpoints and evaluation
models/                               # Saved model files
results/                              # Evaluation reports and visualizations
```

## Common Gotchas

1. **Encoding Issues**: Always use `encoding='gbk'` when reading originaldata.csv
2. **Path Issues**: Many scripts use absolute paths starting with `/Users/yu/code/code2510/gnn/`; update these for different environments
3. **Module Imports**: Some scripts add project root to `sys.path`; ensure this is updated if project structure changes
4. **Device Selection**: Default is CPU; explicitly set `device='cuda'` for GPU training
5. **Edge Building**: Homogeneous graph builder can produce disconnected graphs with certain parameter combinations; always check graph statistics
6. **Threshold Optimization**: Classification threshold optimization (scripts/threshold_optimization.py) should be run after initial training to find optimal decision boundary for F1 score

## Performance Expectations

Based on the project documentation:

| Model         | AUC Range   | F1 Range    | Notes                    |
|---------------|-------------|-------------|--------------------------|
| MLP Baseline  | 0.72-0.78   | 0.35-0.45   | Week 1-2 target          |
| HomoGNN       | 0.76-0.83   | 0.40-0.50   | Week 3-4 target          |
| HeteroGNN++   | 0.82-0.86   | 0.48-0.55   | Future multi-task model  |

## Project Context

This is a research project following a phased implementation plan:
- **Week 1-2**: Data preparation and MLP baseline (COMPLETED)
- **Week 3-4**: Homogeneous GNN implementation (IN PROGRESS)
- **Week 5-6**: Analysis, ablation studies, and technical reports (PLANNED)
- **Future**: Heterogeneous GNN with multi-task learning (PLANNED)

For detailed weekly implementation guides, refer to:
- `Week1_2_Quick_Start.md`
- `Week3_4_Implementation_Guide.md`
- `FINAL_HGNN_Feasibility_Report.md`
