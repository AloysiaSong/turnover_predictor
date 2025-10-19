"""
GCN训练器（增强版）
==================
核心改进:
1. 早停监控Val_AUCPR
2. 完整的评估指标（AUPR, AUROC, F1, Acc等）
3. 阈值扫描与选择
4. 无数据泄漏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算完整评估指标
    
    参数:
        y_true: 真实标签 [N]
        y_prob: 预测概率 [N]
        threshold: 二分类阈值
        
    返回:
        metrics: 指标字典
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        # 概率指标
        'aupr': average_precision_score(y_true, y_prob),
        'auroc': roc_auc_score(y_true, y_prob),
        
        # 二分类指标
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        
        # 混淆矩阵
        'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0
    }
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    except:
        pass
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    在验证集上寻找最优阈值
    
    参数:
        y_true: 真实标签
        y_prob: 预测概率
        metric: 优化目标 {'f1', 'recall', 'precision'}
        thresholds: 候选阈值列表
        
    返回:
        best_threshold: 最优阈值
        best_score: 最优分数
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.75, 0.02)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score


class GCNTrainer:
    """
    GCN训练器
    
    参数:
        model: GCN模型
        data: PyG Data对象
        device: 设备
        lr: 学习率
        weight_decay: L2正则化
        pos_weight: 正样本权重（自动计算或手动指定）
    """
    
    def __init__(
        self,
        model: nn.Module,
        data,
        device: str = 'cpu',
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        pos_weight: Optional[float] = None
    ):
        self.model = model.to(device)
        self.data = data
        self.device = device
        
        # 自动计算pos_weight
        if pos_weight is None:
            num_pos = data.y[data.train_mask].sum().item()
            num_neg = data.train_mask.sum().item() - num_pos
            pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        self.pos_weight = torch.tensor([pos_weight], device=device)
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 学习率调度
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控Val_AUPR（越大越好）
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 早停状态
        self.best_val_aupr = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        out = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
        )
        
        # 计算损失（仅训练集）
        loss = self.criterion(
            out[self.data.train_mask].squeeze(),
            self.data.y[self.data.train_mask].float()
        )
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor) -> Dict[str, float]:
        """
        评估模型
        
        参数:
            mask: 数据集mask
            
        返回:
            metrics: 评估指标
        """
        self.model.eval()
        
        # 预测
        out = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
        )
        
        logits = out[mask].squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()
        
        # 计算指标（默认阈值0.5）
        metrics = compute_metrics(labels, probs, threshold=0.5)
        
        return metrics
    
    def train(
        self,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        save_dir: str = 'outputs/models',
        verbose: bool = True
    ) -> Dict:
        """
        完整训练流程
        
        参数:
            epochs: 最大训练轮数
            early_stopping_patience: 早停耐心
            save_dir: 模型保存目录
            verbose: 是否打印进度
            
        返回:
            history: 训练历史
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_aupr': [],
            'val_auroc': [],
            'val_f1': [],
            'lr': []
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 开始训练（早停监控: Val_AUPR）")
            print(f"{'='*70}")
            print(f"   Epochs: {epochs}")
            print(f"   Early Stopping Patience: {early_stopping_patience}")
            print(f"   pos_weight: {self.pos_weight.item():.2f}")
        
        pbar = tqdm(range(epochs), desc='Training') if verbose else range(epochs)
        
        for epoch in pbar:
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_metrics = self.evaluate(self.data.val_mask)
            val_aupr = val_metrics['aupr']
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_aupr'].append(val_aupr)
            history['val_auroc'].append(val_metrics['auroc'])
            history['val_f1'].append(val_metrics['f1'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 更新进度条
            if verbose:
                pbar.set_postfix({
                    'T_Loss': f"{train_loss:.4f}",
                    'V_AUPR': f"{val_aupr:.4f}",
                    'V_F1': f"{val_metrics['f1']:.4f}"
                })
            
            # 早停检查（基于Val_AUPR）
            if val_aupr > self.best_val_aupr:
                self.best_val_aupr = val_aupr
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_aupr': val_aupr,
                    'val_metrics': val_metrics
                }, f"{save_dir}/best_model.pt")
                
                if verbose and epoch > 0:
                    print(f"\n   ⭐ Epoch {epoch+1}: 新的最佳模型! Val_AUPR={val_aupr:.4f}")
            else:
                self.patience_counter += 1
            
            # 学习率调度
            self.scheduler.step(val_aupr)
            
            # 早停
            if self.patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\n   🛑 Early stopping at epoch {epoch+1}")
                    print(f"   📌 Best epoch: {self.best_epoch+1}, Val_AUPR={self.best_val_aupr:.4f}")
                break
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ 训练完成")
            print(f"{'='*70}")
            print(f"   最佳Epoch: {self.best_epoch+1}")
            print(f"   最佳Val_AUPR: {self.best_val_aupr:.4f}")
        
        # 保存训练历史
        with open(f"{save_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    @torch.no_grad()
    def predict_proba(self, mask: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测概率
        
        参数:
            mask: 数据集mask（默认所有节点）
            
        返回:
            probs: 预测概率
            labels: 真实标签
        """
        self.model.eval()
        
        out = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
        )
        
        if mask is None:
            mask = torch.ones(len(out), dtype=torch.bool, device=out.device)
        
        logits = out[mask].squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()
        
        return probs, labels


def quick_train(
    model: nn.Module,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    early_stopping_patience: int = 20,
    device: str = 'cpu',
    save_dir: str = 'outputs/models'
) -> Tuple[GCNTrainer, Dict]:
    """
    快速训练函数
    
    返回:
        trainer: 训练器对象
        history: 训练历史
    """
    trainer = GCNTrainer(
        model=model,
        data=data,
        device=device,
        lr=lr,
        weight_decay=weight_decay
    )
    
    history = trainer.train(
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir,
        verbose=True
    )
    
    return trainer, history
