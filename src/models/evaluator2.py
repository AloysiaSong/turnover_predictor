"""
GCN评估器

职责:
1. 完整性能评估
2. 计算各种指标
3. 生成混淆矩阵
4. 绘制ROC/PR曲线
5. 与基线模型对比
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from pathlib import Path
import json


class GCNEvaluator:
    """GCN评估器"""
    
    def __init__(self, model, data, device='cpu'):
        """
        Args:
            model: 训练好的GCN模型
            data: PyG Data对象
            device: 设备
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
    @torch.no_grad()
    def predict(self, mask):
        """
        获取预测结果
        
        Args:
            mask: 布尔mask
            
        Returns:
            probs: 预测概率
            preds: 预测标签
            labels: 真实标签
        """
        self.model.eval()
        
        # 前向传播
        out = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
        )
        
        # 获取概率和预测
        probs = torch.sigmoid(out[mask]).squeeze().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        labels = self.data.y[mask].cpu().numpy()
        
        return probs, preds, labels
    
    def evaluate(self, mask, mask_name='Test'):
        """
        完整评估
        
        Args:
            mask: 布尔mask
            mask_name: mask名称 (用于显示)
            
        Returns:
            metrics: 指标字典
        """
        # 获取预测
        probs, preds, labels = self.predict(mask)
        
        # 计算所有指标
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, zero_division=0)
        metrics['recall'] = recall_score(labels, preds, zero_division=0)
        metrics['f1'] = f1_score(labels, preds, zero_division=0)
        
        # AUC指标
        if len(np.unique(labels)) > 1:
            metrics['roc_auc'] = roc_auc_score(labels, probs)
            metrics['pr_auc'] = average_precision_score(labels, probs)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # 分类报告
        metrics['classification_report'] = classification_report(
            labels, preds,
            target_names=['Stay', 'Turnover'],
            zero_division=0
        )
        
        return metrics
    
    def print_metrics(self, metrics, mask_name='Test'):
        """打印评估指标"""
        print("\n" + "="*70)
        print(f"📊 {mask_name}集评估报告")
        print("="*70)
        
        print(f"\n核心指标:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   Accuracy:   {metrics['accuracy']:.4f}")
        print(f"   Precision:  {metrics['precision']:.4f}")
        print(f"   Recall:     {metrics['recall']:.4f}")
        print(f"   F1-Score:   {metrics['f1']:.4f}")
        print(f"   AUC-ROC:    {metrics['roc_auc']:.4f}")
        print(f"   AUC-PR:     {metrics['pr_auc']:.4f}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        print(f"\n混淆矩阵:")
        cm = metrics['confusion_matrix']
        print(f"                预测不离职    预测离职")
        print(f"  实际不离职    {cm[0,0]:>8}      {cm[0,1]:>8}")
        print(f"  实际离职      {cm[1,0]:>8}      {cm[1,1]:>8}")
        
        print(f"\n详细报告:")
        print(metrics['classification_report'])
    
    def plot_confusion_matrix(self, metrics, save_path=None, mask_name='Test'):
        """绘制混淆矩阵"""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Stay', 'Turnover'],
            yticklabels=['Stay', 'Turnover']
        )
        plt.title(f'Confusion Matrix - {mask_name} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ 混淆矩阵已保存: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, mask, save_path=None, mask_name='Test'):
        """绘制ROC曲线"""
        probs, preds, labels = self.predict(mask)
        
        if len(np.unique(labels)) <= 1:
            print("   ⚠️ 只有一个类别，跳过ROC曲线")
            return
        
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'GCN (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {mask_name} Set')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ ROC曲线已保存: {save_path}")
        
        plt.close()
    
    def plot_pr_curve(self, mask, save_path=None, mask_name='Test'):
        """绘制Precision-Recall曲线"""
        probs, preds, labels = self.predict(mask)
        
        if len(np.unique(labels)) <= 1:
            print("   ⚠️ 只有一个类别，跳过PR曲线")
            return
        
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        auc_pr = average_precision_score(labels, probs)
        
        # 计算基线（随机分类器）
        baseline = labels.mean()
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'GCN (AUC = {auc_pr:.3f})', linewidth=2)
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Baseline (AUC = {baseline:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {mask_name} Set')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ PR曲线已保存: {save_path}")
        
        plt.close()
    
    def plot_training_curves(self, history, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss曲线
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Accuracy曲线
        ax = axes[0, 1]
        ax.plot(history['train_acc'], label='Train Acc', linewidth=2)
        ax.plot(history['val_acc'], label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # F1曲线
        ax = axes[1, 0]
        ax.plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-Score')
        ax.set_title('Validation F1-Score')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 学习率曲线
        ax = axes[1, 1]
        ax.plot(history['lr'], label='Learning Rate', linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ 训练曲线已保存: {save_path}")
        
        plt.close()
    
    def full_evaluation(self, save_dir='outputs/evaluation'):
        """
        完整评估流程
        
        Args:
            save_dir: 结果保存目录
            
        Returns:
            results: 所有结果字典
        """
        print("\n" + "="*70)
        print("🎯 开始完整评估")
        print("="*70)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 评估各个集合
        for mask_name, mask in [
            ('Train', self.data.train_mask),
            ('Val', self.data.val_mask),
            ('Test', self.data.test_mask)
        ]:
            print(f"\n{'='*70}")
            print(f"评估{mask_name}集...")
            print(f"{'='*70}")
            
            metrics = self.evaluate(mask, mask_name)
            self.print_metrics(metrics, mask_name)
            results[mask_name.lower()] = metrics
            
            # 生成可视化
            self.plot_confusion_matrix(
                metrics,
                save_path=save_dir / f'{mask_name.lower()}_confusion_matrix.png',
                mask_name=mask_name
            )
            
            self.plot_roc_curve(
                mask,
                save_path=save_dir / f'{mask_name.lower()}_roc_curve.png',
                mask_name=mask_name
            )
            
            self.plot_pr_curve(
                mask,
                save_path=save_dir / f'{mask_name.lower()}_pr_curve.png',
                mask_name=mask_name
            )
        
        # 保存结果
        results_for_save = {}
        for split in ['train', 'val', 'test']:
            results_for_save[split] = {
                'accuracy': float(results[split]['accuracy']),
                'precision': float(results[split]['precision']),
                'recall': float(results[split]['recall']),
                'f1': float(results[split]['f1']),
                'roc_auc': float(results[split]['roc_auc']),
                'pr_auc': float(results[split]['pr_auc'])
            }
        
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(results_for_save, f, indent=2)
        
        print(f"\n{'='*70}")
        print("✅ 评估完成")
        print(f"{'='*70}")
        print(f"\n📁 结果已保存到: {save_dir}")
        
        return results


def compare_with_baseline(gcn_metrics, baseline_metrics):
    """
    与基线模型对比
    
    Args:
        gcn_metrics: GCN指标
        baseline_metrics: 基线指标 (如RandomForest)
    """
    print("\n" + "="*70)
    print("📊 GCN vs 基线模型对比")
    print("="*70)
    
    print(f"\n{'指标':<15} {'GCN':<12} {'基线':<12} {'提升':<12} {'状态'}")
    print("━"*70)
    
    metrics_to_compare = ['f1', 'roc_auc', 'pr_auc', 'recall', 'precision']
    
    for metric in metrics_to_compare:
        gcn_val = gcn_metrics.get(metric, 0)
        base_val = baseline_metrics.get(metric, 0)
        
        if base_val > 0:
            improvement = ((gcn_val - base_val) / base_val) * 100
            status = "✅" if improvement > 0 else "⚠️"
        else:
            improvement = 0
            status = "➖"
        
        print(f"{metric.upper():<15} {gcn_val:<12.4f} {base_val:<12.4f} "
              f"{improvement:>+10.2f}%  {status}")
    
    print("━"*70)


if __name__ == '__main__':
    """测试评估器"""
    print("\n" + "="*70)
    print("🧪 GCN评估器测试")
    print("="*70)
    
    # 加载模型和数据
    print("\n1. 加载模型和数据...")
    data = torch.load('data/processed/homo_graph.pt')
    
    from gcn import create_gcn_model
    model = create_gcn_model(data.num_node_features, architecture='default')
    
    # 尝试加载训练好的模型
    try:
        checkpoint = torch.load('outputs/models/best_model.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✓ 加载训练好的模型")
    except:
        print("   ⚠️ 未找到训练好的模型，使用随机初始化")
    
    # 创建评估器
    print("\n2. 创建评估器...")
    evaluator = GCNEvaluator(model, data, device='cpu')
    
    # 完整评估
    print("\n3. 开始完整评估...")
    results = evaluator.full_evaluation(save_dir='outputs/evaluation')
    
    print("\n" + "="*70)
    print("✅ 评估器测试完成")
    print("="*70)
