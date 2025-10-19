"""
模型训练器
"""
import numpy as np
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Trainer:
    """MLP模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 pos_weight: float = 7.9):
        """
        Args:
            model: PyTorch模型
            device: 设备 (None表示自动选择)
            learning_rate: 学习率
            weight_decay: L2正则化系数
            pos_weight: 正样本权重（用于处理类别不平衡）
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = model.to(device)
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 损失函数（加权BCE）
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        print(f"\n{'='*60}")
        print("训练器初始化")
        print(f"{'='*60}")
        print(f"设备: {self.device}")
        print(f"学习率: {learning_rate}")
        print(f"权重衰减: {weight_decay}")
        print(f"正样本权重: {pos_weight}")
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(batch_x).squeeze()
            loss = self.criterion(logits, batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}'
            })
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader, return_predictions=False):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            logits = self.model(batch_x).squeeze()
            loss = self.criterion(logits, batch_y)
            
            total_loss += loss.item()
            
            # 预测
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        if return_predictions:
            # 合并所有batch
            all_probs = np.concatenate(all_probs)
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            return avg_loss, (all_probs, all_preds, all_labels)
        else:
            return avg_loss
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            early_stopping_patience: int = 15,
            save_dir: str = 'models/mlp',
            verbose: bool = True):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            save_dir: 模型保存目录
            verbose: 是否打印详细信息
        
        Returns:
            history: 训练历史
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print("\n" + "="*60)
            print("开始训练")
            print("="*60)
            print(f"训练集批次: {len(train_loader)}")
            print(f"验证集批次: {len(val_loader)}")
            print(f"最大轮数: {epochs}")
            print(f"早停耐心: {early_stopping_patience}")
            print(f"保存目录: {save_dir}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.evaluate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            epoch_time = time.time() - start_time
            
            # 打印进度
            if verbose:
                print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, save_dir / 'best_model.pt')
                
                if verbose:
                    print(f"  ✅ 保存最佳模型 (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if verbose:
                    print(f"  ⏳ Early stopping: {patience_counter}/{early_stopping_patience}")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\n⛔ Early stopping at epoch {epoch}")
                break
        
        # 加载最佳模型
        checkpoint = torch.load(save_dir / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 保存训练历史
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        if verbose:
            print("\n✅ 训练完成！")
            print(f"   最佳验证损失: {best_val_loss:.4f}")
            print(f"   训练轮数: {epoch}")
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✅ 加载checkpoint: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint


def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    """
    计算二分类评估指标
    
    Args:
        y_true: 真实标签 (array-like)
        y_prob: 预测概率 (array-like)
        threshold: 二分类阈值
    
    Returns:
        指标字典，包含AUPR、AUROC、Accuracy、Precision、Recall、F1以及混淆矩阵元素
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'aupr': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'auroc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tn': 0,
        'fp': 0,
        'fn': 0,
        'tp': 0,
    }
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    except ValueError:
        # confusion_matrix 可能因为单类数据而失败，已通过zero_division处理
        pass
    
    return metrics


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size=32, num_workers=0):
    """
    创建数据加载器
    
    Args:
        X_train, y_train: 训练集特征和标签
        X_val, y_val: 验证集特征和标签
        X_test, y_test: 测试集特征和标签
        batch_size: 批大小
        num_workers: 工作线程数
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    # 创建加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\n{'='*60}")
    print("数据加载器创建成功")
    print(f"{'='*60}")
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    print(f"批大小: {batch_size}")
    
    return train_loader, val_loader, test_loader
