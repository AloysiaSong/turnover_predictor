"""
GCN完整训练脚本

一键运行完整流程:
1. 加载数据
2. 创建模型
3. 训练模型
4. 评估性能
5. 生成报告
"""

import torch
import sys
from pathlib import Path


# 添加src/models到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'models'))

from gcn import create_gcn_model
from trainer2 import quick_train
from evaluator2 import GCNEvaluator

import json


def main():
    """主训练流程"""
    print("\n" + "="*70)
    print("🚀 GCN完整训练流程")
    print("="*70)
    
    # ========== 配置 ==========
    config = {
        'data_path': 'data/processed/homo_graph.pt',
        'architecture': 'default',  # shallow, default, deep, very_deep
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'epochs': 200,
        'early_stopping_patience': 20,
        'device': 'cpu',  # 'cuda' if torch.cuda.is_available() else 'cpu'
        'save_dir': 'outputs/models',
        'eval_dir': 'outputs/evaluation'
    }
    
    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # ========== Step 1: 加载数据 ==========
    print("\n" + "="*70)
    print("📂 Step 1/5: 加载数据")
    print("="*70)
    
    data = torch.load(config['data_path'])
    print(f"\n   ✓ 数据加载成功")
    print(f"   ✓ 节点数: {data.num_nodes}")
    print(f"   ✓ 边数: {data.num_edges}")
    print(f"   ✓ 特征维度: {data.num_node_features}")
    print(f"   ✓ 训练集: {data.train_mask.sum()}")
    print(f"   ✓ 验证集: {data.val_mask.sum()}")
    print(f"   ✓ 测试集: {data.test_mask.sum()}")
    
    # ========== Step 2: 创建模型 ==========
    print("\n" + "="*70)
    print("🏗️ Step 2/5: 创建模型")
    print("="*70)
    
    model = create_gcn_model(
        in_channels=data.num_node_features,
        architecture=config['architecture'],
        dropout=config['dropout']
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n   ✓ 模型创建成功")
    print(f"   ✓ 架构: {config['architecture']}")
    print(f"   ✓ 参数量: {num_params:,}")
    
    # ========== Step 3: 训练模型 ==========
    print("\n" + "="*70)
    print("🎯 Step 3/5: 训练模型")
    print("="*70)
    
    trainer, history = quick_train(
        model=model,
        data=data,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        early_stopping_patience=config['early_stopping_patience'],
        device=config['device'],
        save_dir=config['save_dir']
    )
    
    # ========== Step 4: 测试集评估 ==========
    print("\n" + "="*70)
    print("📊 Step 4/5: 测试集快速评估")
    print("="*70)
    
    test_loss, test_acc, test_f1 = trainer.evaluate(data.test_mask)
    print(f"\n   测试Loss: {test_loss:.4f}")
    print(f"   测试Acc: {test_acc:.4f}")
    print(f"   测试F1: {test_f1:.4f}")
    
    # ========== Step 5: 完整评估 ==========
    print("\n" + "="*70)
    print("🎨 Step 5/5: 完整评估与可视化")
    print("="*70)
    
    evaluator = GCNEvaluator(model, data, device=config['device'])
    results = evaluator.full_evaluation(save_dir=config['eval_dir'])
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    evaluator.plot_training_curves(
        history,
        save_path=f"{config['eval_dir']}/training_curves.png"
    )
    
    # ========== 生成总结报告 ==========
    print("\n" + "="*70)
    print("📄 生成总结报告")
    print("="*70)
    
    report = {
        'config': config,
        'training': {
            'best_epoch': trainer.best_epoch + 1,
            'best_val_loss': trainer.best_val_loss,
            'best_val_f1': max(history['val_f1']),
            'total_epochs': len(history['train_loss'])
        },
        'test_results': {
            'accuracy': float(results['test']['accuracy']),
            'precision': float(results['test']['precision']),
            'recall': float(results['test']['recall']),
            'f1': float(results['test']['f1']),
            'roc_auc': float(results['test']['roc_auc']),
            'pr_auc': float(results['test']['pr_auc'])
        }
    }
    
    report_path = Path(config['eval_dir']) / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n   ✓ 报告已保存: {report_path}")
    
    # ========== 最终总结 ==========
    print("\n" + "="*70)
    print("✅ 训练流程完成！")
    print("="*70)
    
    print("\n📊 最终结果:")
    print(f"   测试Accuracy: {results['test']['accuracy']:.4f}")
    print(f"   测试Precision: {results['test']['precision']:.4f}")
    print(f"   测试Recall: {results['test']['recall']:.4f}")
    print(f"   测试F1-Score: {results['test']['f1']:.4f}")
    print(f"   测试AUC-ROC: {results['test']['roc_auc']:.4f}")
    print(f"   测试AUC-PR: {results['test']['pr_auc']:.4f}")
    
    print("\n📁 生成文件:")
    print(f"   模型: {config['save_dir']}/best_model.pt")
    print(f"   训练历史: {config['save_dir']}/training_history.json")
    print(f"   评估结果: {config['eval_dir']}/metrics.json")
    print(f"   训练曲线: {config['eval_dir']}/training_curves.png")
    print(f"   混淆矩阵: {config['eval_dir']}/test_confusion_matrix.png")
    print(f"   ROC曲线: {config['eval_dir']}/test_roc_curve.png")
    print(f"   PR曲线: {config['eval_dir']}/test_pr_curve.png")
    print(f"   总结报告: {config['eval_dir']}/training_report.json")
    
    print("\n" + "="*70)
    print("🎉 恭喜！GCN训练完成！")
    print("="*70)
    
    return model, results


if __name__ == '__main__':
    model, results = main()
