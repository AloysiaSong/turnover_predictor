"""
分类阈值优化工具
找到最佳的分类阈值以平衡Precision和Recall
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
import json
from pathlib import Path


class ThresholdOptimizer:
    """分类阈值优化器"""
    
    def __init__(self, y_true, y_prob):
        """
        Args:
            y_true: 真实标签
            y_prob: 预测概率
        """
        self.y_true = y_true
        self.y_prob = y_prob
        
    def find_best_threshold(self, metric='f1'):
        """
        寻找最佳阈值
        
        Args:
            metric: 优化指标 ('f1', 'balanced', 'youden')
            
        Returns:
            best_threshold, best_score, results
        """
        # 扩大搜索范围，使用更细的步长
        thresholds = np.arange(0.05, 0.95, 0.01)
        
        results = {
            'thresholds': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': []
        }
        
        for threshold in thresholds:
            y_pred = (self.y_prob >= threshold).astype(int)
            
            # 跳过所有预测为同一类的情况
            if len(np.unique(y_pred)) < 2:
                continue
            
            precision = precision_score(self.y_true, y_pred, zero_division=0)
            recall = recall_score(self.y_true, y_pred, zero_division=0)
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            accuracy = (self.y_true == y_pred).mean()
            
            results['thresholds'].append(threshold)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['accuracy'].append(accuracy)
        
        # 根据指标选择最佳阈值
        if metric == 'f1':
            idx = np.argmax(results['f1'])
            best_score = results['f1'][idx]
        elif metric == 'balanced':
            # 平衡Precision和Recall
            scores = [p * r for p, r in zip(results['precision'], results['recall'])]
            idx = np.argmax(scores)
            best_score = scores[idx]
        elif metric == 'youden':
            # Youden's J statistic
            scores = [r - (1 - p) for p, r in zip(results['precision'], results['recall'])]
            idx = np.argmax(scores)
            best_score = scores[idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        best_threshold = results['thresholds'][idx]
        
        return best_threshold, best_score, results
    
    def plot_threshold_analysis(self, save_path='results/mlp/threshold_analysis.png'):
        """绘制阈值分析图"""
        _, _, results = self.find_best_threshold('f1')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Precision vs Threshold
        axes[0, 0].plot(results['thresholds'], results['precision'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Threshold', fontsize=11)
        axes[0, 0].set_ylabel('Precision', fontsize=11)
        axes[0, 0].set_title('Precision vs Threshold', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(0.5, color='r', linestyle='--', label='Default (0.5)')
        axes[0, 0].legend()
        
        # 2. Recall vs Threshold
        axes[0, 1].plot(results['thresholds'], results['recall'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Threshold', fontsize=11)
        axes[0, 1].set_ylabel('Recall', fontsize=11)
        axes[0, 1].set_title('Recall vs Threshold', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(0.5, color='r', linestyle='--', label='Default (0.5)')
        axes[0, 1].legend()
        
        # 3. F1 Score vs Threshold
        axes[1, 0].plot(results['thresholds'], results['f1'], 'orange', linewidth=2)
        best_f1_idx = np.argmax(results['f1'])
        best_f1_threshold = results['thresholds'][best_f1_idx]
        best_f1 = results['f1'][best_f1_idx]
        axes[1, 0].scatter([best_f1_threshold], [best_f1], color='red', s=100, zorder=5)
        axes[1, 0].set_xlabel('Threshold', fontsize=11)
        axes[1, 0].set_ylabel('F1 Score', fontsize=11)
        axes[1, 0].set_title(f'F1 Score vs Threshold (Best={best_f1_threshold:.2f})', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(0.5, color='r', linestyle='--', label='Default (0.5)')
        axes[1, 0].legend()
        
        # 4. Precision-Recall Trade-off
        axes[1, 1].plot(results['thresholds'], results['precision'], 'b-', label='Precision', linewidth=2)
        axes[1, 1].plot(results['thresholds'], results['recall'], 'g-', label='Recall', linewidth=2)
        axes[1, 1].plot(results['thresholds'], results['f1'], 'orange', label='F1', linewidth=2)
        axes[1, 1].set_xlabel('Threshold', fontsize=11)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Precision-Recall-F1 Trade-off', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(0.5, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 阈值分析图已保存: {save_path}")
        
        return best_f1_threshold, best_f1
    
    def generate_threshold_report(self, save_dir='results/mlp'):
        """生成阈值优化报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("阈值优化分析")
        print("="*60)
        
        # 不同指标下的最佳阈值
        metrics_to_optimize = ['f1', 'balanced']
        
        results_summary = {}
        
        for metric in metrics_to_optimize:
            best_threshold, best_score, _ = self.find_best_threshold(metric)
            
            # 计算该阈值下的所有指标
            y_pred = (self.y_prob >= best_threshold).astype(int)
            
            precision = precision_score(self.y_true, y_pred, zero_division=0)
            recall = recall_score(self.y_true, y_pred, zero_division=0)
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            accuracy = (self.y_true == y_pred).mean()
            
            results_summary[metric] = {
                'threshold': float(best_threshold),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy)
            }
            
            print(f"\n优化指标: {metric}")
            print(f"  最佳阈值: {best_threshold:.3f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        # 默认阈值0.5的性能
        y_pred_default = (self.y_prob >= 0.5).astype(int)
        default_results = {
            'threshold': 0.5,
            'precision': float(precision_score(self.y_true, y_pred_default, zero_division=0)),
            'recall': float(recall_score(self.y_true, y_pred_default, zero_division=0)),
            'f1': float(f1_score(self.y_true, y_pred_default, zero_division=0)),
            'accuracy': float((self.y_true == y_pred_default).mean())
        }
        results_summary['default'] = default_results
        
        print(f"\n默认阈值: 0.5")
        print(f"  Precision: {default_results['precision']:.4f}")
        print(f"  Recall: {default_results['recall']:.4f}")
        print(f"  F1: {default_results['f1']:.4f}")
        print(f"  Accuracy: {default_results['accuracy']:.4f}")
        
        # 添加建议
        print("\n" + "="*60)
        print("💡 阈值选择建议")
        print("="*60)
        
        f1_threshold = results_summary['f1']['threshold']
        f1_improvement = results_summary['f1']['f1'] - default_results['f1']
        
        print(f"\n1. 如果追求F1最优:")
        print(f"   推荐阈值: {f1_threshold:.3f}")
        print(f"   F1提升: {f1_improvement:+.4f} ({f1_improvement/default_results['f1']*100:+.1f}%)")
        print(f"   权衡: Precision={results_summary['f1']['precision']:.3f}, Recall={results_summary['f1']['recall']:.3f}")
        
        # 找一个平衡的阈值（Precision和Recall较接近）
        best_balanced_idx = None
        min_diff = float('inf')
        _, _, all_results = self.find_best_threshold('f1')
        
        for i, (p, r) in enumerate(zip(all_results['precision'], all_results['recall'])):
            if p > 0.3 and r > 0.5:  # 确保最低标准
                diff = abs(p - r)
                if diff < min_diff:
                    min_diff = diff
                    best_balanced_idx = i
        
        if best_balanced_idx is not None:
            balanced_threshold = all_results['thresholds'][best_balanced_idx]
            balanced_p = all_results['precision'][best_balanced_idx]
            balanced_r = all_results['recall'][best_balanced_idx]
            balanced_f1 = all_results['f1'][best_balanced_idx]
            
            print(f"\n2. 如果追求Precision和Recall平衡:")
            print(f"   推荐阈值: {balanced_threshold:.3f}")
            print(f"   Precision: {balanced_p:.3f}")
            print(f"   Recall: {balanced_r:.3f}")
            print(f"   F1: {balanced_f1:.3f}")
        
        print(f"\n3. 业务场景建议:")
        print(f"   • 重视召回率（不想漏掉离职员工）→ 使用较低阈值（0.3-0.4）")
        print(f"   • 重视精确率（减少误报成本）→ 使用较高阈值（0.5-0.6）")
        print(f"   • 平衡考虑 → 使用F1最优阈值")
        
        # 保存结果
        with open(save_dir / 'threshold_optimization.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 绘制分析图
        best_threshold, best_f1 = self.plot_threshold_analysis(
            save_path=save_dir / 'threshold_analysis.png'
        )
        
        print(f"\n✅ 阈值优化报告已保存: {save_dir}")
        
        return results_summary


def optimize_threshold(y_true, y_prob, save_dir='results/mlp'):
    """
    快速进行阈值优化
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        save_dir: 保存目录
        
    Returns:
        results_summary: 优化结果摘要
    """
    optimizer = ThresholdOptimizer(y_true, y_prob)
    results_summary = optimizer.generate_threshold_report(save_dir)
    
    return results_summary


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    
    results = optimize_threshold(y_true, y_prob)
    print("\n优化结果:")
    print(json.dumps(results, indent=2))