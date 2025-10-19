"""
模型评估器
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
from pathlib import Path
import json


class Evaluator:
    """模型评估器"""
    
    def __init__(self, save_dir='results/mlp'):
        """
        Args:
            save_dir: 结果保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def evaluate(self, y_true, y_pred, y_prob, set_name='Test'):
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            set_name: 数据集名称
            
        Returns:
            metrics: 评估指标字典
        """
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob),
            'ap': average_precision_score(y_true, y_prob)
        }
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"{set_name}集评估结果")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print(f"AP:        {metrics['ap']:.4f}")
        
        # 保存指标
        with open(self.save_dir / f'{set_name.lower()}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, set_name='Test'):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['不离职', '离职'],
                    yticklabels=['不离职', '离职'])
        plt.title(f'{set_name}集混淆矩阵', fontsize=14, pad=15)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()
        
        save_path = self.save_dir / f'{set_name.lower()}_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 混淆矩阵已保存: {save_path}")
    
    def plot_roc_curve(self, y_true, y_prob, set_name='Test'):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{set_name}集 ROC曲线', fontsize=14, pad=15)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / f'{set_name.lower()}_roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC曲线已保存: {save_path}")
    
    def plot_pr_curve(self, y_true, y_prob, set_name='Test'):
        """绘制Precision-Recall曲线"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.4f})', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'{set_name}集 Precision-Recall曲线', fontsize=14, pad=15)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / f'{set_name.lower()}_pr_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PR曲线已保存: {save_path}")
    
    def plot_training_history(self, history):
        """绘制训练历史"""
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        epochs = range(1, len(train_loss) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        plt.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练历史', fontsize=14, pad=15)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练历史已保存: {save_path}")
    
    def generate_report(self, y_true, y_pred, y_prob, set_name='Test'):
        """
        生成完整评估报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            set_name: 数据集名称
        """
        print(f"\n{'='*60}")
        print(f"生成{set_name}集评估报告")
        print(f"{'='*60}")
        
        # 1. 计算指标
        metrics = self.evaluate(y_true, y_pred, y_prob, set_name)
        
        # 2. 绘制图表
        print(f"\n绘制可视化图表...")
        self.plot_confusion_matrix(y_true, y_pred, set_name)
        self.plot_roc_curve(y_true, y_prob, set_name)
        self.plot_pr_curve(y_true, y_prob, set_name)
        
        # 3. 详细分类报告
        print(f"\n{set_name}集详细分类报告:")
        print("-" * 60)
        report = classification_report(
            y_true, y_pred,
            target_names=['不离职', '离职'],
            digits=4
        )
        print(report)
        
        # 保存分类报告
        with open(self.save_dir / f'{set_name.lower()}_classification_report.txt', 'w') as f:
            f.write(report)
        
        # 4. 生成markdown报告
        self._generate_markdown_report(metrics, set_name)
        
        print(f"\n✅ {set_name}集评估完成！")
        print(f"   结果保存在: {self.save_dir}")
        
        return metrics
    
    def _generate_markdown_report(self, metrics, set_name='Test'):
        """生成Markdown格式的报告"""
        report = f"""# {set_name}集评估报告

## 性能指标

| 指标 | 数值 |
|------|------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 Score | {metrics['f1']:.4f} |
| AUC-ROC | {metrics['auc']:.4f} |
| Average Precision | {metrics['ap']:.4f} |

## 可视化结果

### 混淆矩阵
![混淆矩阵]({set_name.lower()}_confusion_matrix.png)

### ROC曲线
![ROC曲线]({set_name.lower()}_roc_curve.png)

### Precision-Recall曲线
![PR曲线]({set_name.lower()}_pr_curve.png)

---
报告生成时间: {np.datetime64('now')}
"""
        
        with open(self.save_dir / f'{set_name.lower()}_report.md', 'w') as f:
            f.write(report)
        
        print(f"✅ Markdown报告已保存")


def evaluate_model(model, test_loader, device='cpu', save_dir='results/mlp'):
    """
    快速评估模型
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        save_dir: 保存目录
        
    Returns:
        metrics: 评估指标
    """
    import torch
    
    model.eval()
    model = model.to(device)
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            # 预测
            logits = model(batch_x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.numpy())
    
    # 合并结果
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    # 评估
    evaluator = Evaluator(save_dir=save_dir)
    metrics = evaluator.generate_report(y_true, y_pred, y_prob, set_name='Test')
    
    return metrics
