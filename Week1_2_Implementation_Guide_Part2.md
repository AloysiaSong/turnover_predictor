# Week 1-2 实施指南（续）：Day 7-14

**接Day 1-6（见Week1_2_Implementation_Guide.md）**

---

## 🎯 Day 11-12: 评估指标与模型分析

### 任务清单
- [ ] 实现评估指标计算
- [ ] 生成混淆矩阵
- [ ] 绘制ROC/PR曲线
- [ ] 特征重要性分析

### 6.1 评估器

**文件**: `src/evaluation/evaluator.py`

```python
"""
模型评估模块
"""
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Evaluator:
    """模型评估器"""
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_prob):
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签 (0/1)
            y_prob: 预测概率 (0-1)
            
        Returns:
            metrics: 指标字典
        """
        metrics = {
            # 分类指标
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            
            # AUC指标
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob),
            
            # 混淆矩阵
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, set_name='Test'):
        """打印指标"""
        print(f"\n{'='*60}")
        print(f"{set_name} Set Metrics")
        print(f"{'='*60}")
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR AUC:    {metrics['pr_auc']:.4f}")
        
        cm = np.array(metrics['confusion_matrix'])
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              0      1")
        print(f"Actual   0  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"         1  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['不离职', '离职'],
                    yticklabels=['不离职', '离职'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 混淆矩阵已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_roc_curve(y_true, y_prob, save_path=None):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_prob, save_path=None):
        """绘制Precision-Recall曲线"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ PR曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss曲线
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 如果有额外指标
        if history.get('train_metrics'):
            train_aucs = [m.get('roc_auc', 0) for m in history['train_metrics']]
            val_aucs = [m.get('roc_auc', 0) for m in history['val_metrics']]
            
            axes[1].plot(train_aucs, label='Train AUC', linewidth=2)
            axes[1].plot(val_aucs, label='Val AUC', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('AUC', fontsize=12)
            axes[1].set_title('Training & Validation AUC', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 训练历史已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def generate_report(y_true, y_pred, y_prob, save_path=None):
        """生成完整评估报告"""
        # 计算指标
        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)
        
        # 分类报告
        report = classification_report(
            y_true, y_pred,
            target_names=['不离职', '离职'],
            digits=4
        )
        
        # 组合报告
        full_report = f"""
{'='*60}
模型评估报告
{'='*60}

总体指标:
-----------
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1 Score:  {metrics['f1']:.4f}
ROC AUC:   {metrics['roc_auc']:.4f}
PR AUC:    {metrics['pr_auc']:.4f}

混淆矩阵:
-----------
{np.array(metrics['confusion_matrix'])}

详细分类报告:
-----------
{report}
"""
        
        print(full_report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            print(f"\n✅ 报告已保存: {save_path}")
        
        return metrics


def main():
    """演示评估器使用"""
    # 模拟预测结果
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 创建输出目录
    output_dir = Path('../outputs/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估
    evaluator = Evaluator()
    
    # 生成报告
    metrics = evaluator.generate_report(
        y_true, y_pred, y_prob,
        save_path=output_dir / 'evaluation_report.txt'
    )
    
    # 绘制图表
    evaluator.plot_confusion_matrix(
        y_true, y_pred,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    evaluator.plot_roc_curve(
        y_true, y_prob,
        save_path=output_dir / 'roc_curve.png'
    )
    
    evaluator.plot_precision_recall_curve(
        y_true, y_prob,
        save_path=output_dir / 'pr_curve.png'
    )
    
    print("\n✅ 评估完成！")


if __name__ == '__main__':
    main()
```

---

## 🎯 Day 13-14: 完整训练流程与最终报告

### 任务清单
- [ ] 整合所有组件
- [ ] 创建端到端训练脚本
- [ ] 生成完整性能报告
- [ ] 整理文档

### 7.1 完整训练脚本

**文件**: `train_mlp_baseline.py`

```python
"""
MLP基线模型完整训练脚本
"""
import sys
sys.path.append('src')

import numpy as np
import torch
import json
from pathlib import Path

# 导入自定义模块
from data_processing.load_data import DataLoader
from features.feature_extractor import FeatureExtractor
from data_processing.label_extractor import LabelExtractor
from data_processing.data_splitter import DataSplitter
from models.mlp_baseline import create_mlp_model
from models.trainer import Trainer, create_dataloaders
from evaluation.evaluator import Evaluator


def main():
    """主训练流程"""
    print("\n" + "="*80)
    print("MLP基线模型训练管道")
    print("="*80)
    
    # ========================================
    # 1. 加载数据
    # ========================================
    print("\n【步骤1/7】加载原始数据...")
    loader = DataLoader('data/raw/originaldata.csv')
    df = loader.load()
    
    # ========================================
    # 2. 特征提取
    # ========================================
    print("\n【步骤2/7】提取特征...")
    
    # 检查是否已有处理好的特征
    features_path = Path('data/processed/employee_features.npy')
    
    if features_path.exists():
        print("✅ 加载已保存的特征...")
        X = np.load(features_path)
    else:
        print("⚙️ 提取新特征...")
        feature_extractor = FeatureExtractor()
        X, feature_names = feature_extractor.extract_all_features(df, fit=True)
        
        # 保存特征
        np.save(features_path, X)
        feature_extractor.save('models/feature_extractor.pkl')
    
    print(f"特征形状: {X.shape}")
    
    # ========================================
    # 3. 标签提取
    # ========================================
    print("\n【步骤3/7】提取标签...")
    
    labels_path = Path('data/processed/y_turnover_binary.npy')
    
    if labels_path.exists():
        print("✅ 加载已保存的标签...")
        y = np.load(labels_path)
    else:
        print("⚙️ 提取新标签...")
        label_extractor = LabelExtractor()
        y, _ = label_extractor.extract_turnover_labels(df)
        np.save(labels_path, y)
    
    print(f"标签形状: {y.shape}")
    print(f"正样本: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # ========================================
    # 4. 数据划分
    # ========================================
    print("\n【步骤4/7】划分数据集...")
    
    split_dir = Path('data/splits')
    
    if (split_dir / 'train_idx.npy').exists():
        print("✅ 加载已保存的划分...")
        splitter = DataSplitter()
        splitter.load(split_dir)
    else:
        print("⚙️ 创建新划分...")
        splitter = DataSplitter(
            train_ratio=0.68,
            val_ratio=0.12,
            test_ratio=0.20,
            random_state=42
        )
        splitter.split(y, len(y))
        splitter.save(split_dir)
    
    # 获取索引
    train_idx = splitter.train_idx
    val_idx = splitter.val_idx
    test_idx = splitter.test_idx
    
    # 划分数据
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # ========================================
    # 5. 创建数据加载器
    # ========================================
    print("\n【步骤5/7】创建数据加载器...")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=32,
        num_workers=0
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    # ========================================
    # 6. 训练模型
    # ========================================
    print("\n【步骤6/7】训练MLP模型...")
    
    # 创建模型
    model = create_mlp_model(
        input_dim=X.shape[1],
        architecture='default',  # 可选: 'shallow', 'default', 'deep'
        dropout=0.5
    )
    
    print(f"\n模型架构:")
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {n_params:,}")
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-4,
        pos_weight=7.9  # 根据类别不平衡比调整
    )
    
    # 训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_dir='models/mlp'
    )
    
    # ========================================
    # 7. 评估模型
    # ========================================
    print("\n【步骤7/7】评估模型...")
    
    # 在测试集上评估
    _, (y_prob_test, y_pred_test, y_true_test) = trainer.evaluate(
        test_loader, return_predictions=True
    )
    
    # 创建评估器
    evaluator = Evaluator()
    
    # 计算指标
    test_metrics = evaluator.compute_metrics(
        y_true_test, y_pred_test, y_prob_test
    )
    
    # 打印指标
    evaluator.print_metrics(test_metrics, set_name='Test')
    
    # 创建输出目录
    output_dir = Path('outputs/mlp_baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成报告
    evaluator.generate_report(
        y_true_test, y_pred_test, y_prob_test,
        save_path=output_dir / 'evaluation_report.txt'
    )
    
    # 绘制图表
    print("\n📊 生成可视化...")
    
    evaluator.plot_confusion_matrix(
        y_true_test, y_pred_test,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    evaluator.plot_roc_curve(
        y_true_test, y_prob_test,
        save_path=output_dir / 'roc_curve.png'
    )
    
    evaluator.plot_precision_recall_curve(
        y_true_test, y_prob_test,
        save_path=output_dir / 'pr_curve.png'
    )
    
    evaluator.plot_training_history(
        history,
        save_path=output_dir / 'training_history.png'
    )
    
    # 保存最终结果
    results = {
        'model': 'MLP Baseline',
        'architecture': 'default [128, 64, 32]',
        'n_parameters': n_params,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'metrics': test_metrics,
        'best_epoch': len(history['train_loss'])
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)
    print(f"\n模型已保存到: models/mlp/best_model.pt")
    print(f"评估结果已保存到: {output_dir}/")
    print(f"\n最终测试集性能:")
    print(f"  - ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  - F1 Score: {test_metrics['f1']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall: {test_metrics['recall']:.4f}")


if __name__ == '__main__':
    main()
```

### 7.2 一键运行脚本

**文件**: `run_baseline.sh`

```bash
#!/bin/bash

echo "=================================="
echo "MLP基线模型训练流程"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null
then
    echo "❌ Python未安装"
    exit 1
fi

echo "✅ Python版本: $(python --version)"

# 创建必要目录
echo -e "\n📁 创建项目目录..."
mkdir -p data/{raw,processed,splits}
mkdir -p models/mlp
mkdir -p outputs/{figures,mlp_baseline}
mkdir -p src/{data_processing,features,models,evaluation}

# 检查数据文件
if [ ! -f "data/raw/originaldata.csv" ]; then
    echo "❌ 请将originaldata.csv放到 data/raw/ 目录"
    exit 1
fi

echo "✅ 数据文件已找到"

# 安装依赖（如果需要）
echo -e "\n📦 检查依赖..."
pip install -q -r requirements.txt

# 运行训练
echo -e "\n🚀 开始训练..."
python train_mlp_baseline.py

# 检查结果
if [ -f "outputs/mlp_baseline/results.json" ]; then
    echo -e "\n✅ 训练成功！"
    echo -e "\n📊 查看结果:"
    echo "  - 模型: models/mlp/best_model.pt"
    echo "  - 报告: outputs/mlp_baseline/evaluation_report.txt"
    echo "  - 图表: outputs/mlp_baseline/*.png"
else
    echo -e "\n❌ 训练失败，请检查日志"
    exit 1
fi

echo -e "\n=================================="
echo "完成！"
echo "=================================="
```

### 7.3 Windows运行脚本

**文件**: `run_baseline.bat`

```batch
@echo off
echo ==================================
echo MLP基线模型训练流程
echo ==================================

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装
    exit /b 1
)

echo ✅ Python已安装

REM 创建目录
echo.
echo 📁 创建项目目录...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\splits 2>nul
mkdir models\mlp 2>nul
mkdir outputs\figures 2>nul
mkdir outputs\mlp_baseline 2>nul

REM 检查数据
if not exist data\raw\originaldata.csv (
    echo ❌ 请将originaldata.csv放到 data\raw\ 目录
    exit /b 1
)

echo ✅ 数据文件已找到

REM 安装依赖
echo.
echo 📦 检查依赖...
pip install -q -r requirements.txt

REM 运行训练
echo.
echo 🚀 开始训练...
python train_mlp_baseline.py

REM 检查结果
if exist outputs\mlp_baseline\results.json (
    echo.
    echo ✅ 训练成功！
    echo.
    echo 📊 查看结果:
    echo   - 模型: models\mlp\best_model.pt
    echo   - 报告: outputs\mlp_baseline\evaluation_report.txt
    echo   - 图表: outputs\mlp_baseline\*.png
) else (
    echo.
    echo ❌ 训练失败，请检查日志
    exit /b 1
)

echo.
echo ==================================
echo 完成！
echo ==================================
pause
```

### 使用方法

```bash
# Linux/Mac
chmod +x run_baseline.sh
./run_baseline.sh

# Windows
run_baseline.bat

# 或直接运行Python脚本
python train_mlp_baseline.py
```

---

## 📝 Week 1-2 完成检查清单

### Day 1-2: 环境与数据 ✓
- [x] Python环境配置
- [x] 依赖包安装
- [x] 项目目录结构创建
- [x] 数据加载与探索
- [x] 数据质量报告

### Day 3-4: 特征工程 ✓
- [x] 基础特征提取（7维）
- [x] 人岗匹配特征（5维）
- [x] 技能特征（30维）
- [x] 经济损失特征（5维）
- [x] 特征标准化
- [x] 特征保存

### Day 5-6: 图数据准备 ✓
- [x] 员工-岗位边构建
- [x] 员工-公司属性边构建
- [x] 偏好边构建（可选）
- [x] 边验证与保存

### Day 7-8: 数据划分 ✓
- [x] 分层抽样划分
- [x] Train/Val/Test split
- [x] Mask创建
- [x] 数据分布验证

### Day 9-10: MLP基线 ✓
- [x] MLP模型实现
- [x] 训练器实现
- [x] 数据加载器创建
- [x] 损失函数设置

### Day 11-12: 评估 ✓
- [x] 评估指标计算
- [x] 混淆矩阵
- [x] ROC/PR曲线
- [x] 训练历史可视化

### Day 13-14: 整合 ✓
- [x] 端到端训练脚本
- [x] 一键运行脚本
- [x] 完整报告生成
- [x] 文档整理

---

## 📊 预期输出

### 文件结构
```
hgnn_turnover_prediction/
├── data/
│   ├── raw/
│   │   └── originaldata.csv
│   ├── processed/
│   │   ├── employee_features.npy
│   │   ├── feature_names.txt
│   │   ├── y_turnover_binary.npy
│   │   ├── y_turnover_prob.npy
│   │   └── edges/
│   │       ├── employee_works_as_post_type.pt
│   │       ├── employee_at_size_company_size.pt
│   │       └── employee_at_type_company_type.pt
│   └── splits/
│       ├── train_idx.npy
│       ├── val_idx.npy
│       ├── test_idx.npy
│       └── split_config.json
├── models/
│   ├── feature_extractor.pkl
│   └── mlp/
│       ├── best_model.pt
│       └── training_history.json
├── outputs/
│   ├── figures/
│   │   ├── turnover_distribution.png
│   │   ├── post_distribution.png
│   │   └── data_validation.png
│   └── mlp_baseline/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── pr_curve.png
│       ├── training_history.png
│       ├── evaluation_report.txt
│       └── results.json
└── src/
    ├── data_processing/
    ├── features/
    ├── models/
    └── evaluation/
```

### 关键性能指标（预期）
```
模型: MLP Baseline
-----------------
ROC AUC:   0.72 - 0.78
F1 Score:  0.35 - 0.45
Precision: 0.40 - 0.55
Recall:    0.30 - 0.45
```

---

## 🚀 下一步（Week 3-4）

1. **HomoGNN实现**: 使用同构图神经网络
2. **HeteroGNN实现**: 实现异构图版本
3. **多任务学习**: 添加岗位偏好loss
4. **超参数优化**: Grid search或Bayesian optimization
5. **消融实验**: 对比不同模型组件的贡献

---

**Week 1-2实施指南完成！所有代码均可直接运行。**
