"""
GCN训练器

职责:
1. 管理完整训练流程
2. 优化器和学习率调度
3. 早停机制
4. 模型保存和加载
5. 训练日志记录
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import time
import json
from tqdm import tqdm


class GCNTrainer:
    """GCN训练器"""
    
    def __init__(
        self,
        model,
        data,
        device='cpu',
        lr=0.01,
        weight_decay=5e-4,
        pos_weight=None,
        scheduler_patience=10,
        scheduler_factor=0.5,
        min_lr=1e-6
    ):
        """
        Args:
            model: GCN模型
            data: PyG Data对象
            device: 设备 ('cpu' or 'cuda')
            lr: 初始学习率
            weight_decay: L2正则化系数
            pos_weight: 正样本权重（处理类别不平衡）
            scheduler_patience: 学习率调度器耐心值
            scheduler_factor: 学习率衰减因子
            min_lr: 最小学习率
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 损失函数（带类别权重）
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
            verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }
        
        # 最佳模型追踪
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        # 前向传播
        out = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
        )
        
        # 计算损失（只在训练集上）
        loss = self.criterion(
            out[self.data.train_mask].squeeze(),
            self.data.y[self.data.train_mask].float()
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 计算训练准确率
        with torch.no_grad():
            probs = torch.sigmoid(out[self.data.train_mask]).squeeze()
            preds = (probs > 0.5).long()
            acc = (preds == self.data.y[self.data.train_mask]).float().mean()
        
        return loss.item(), acc.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """
        在指定mask上评估
        
        Args:
            mask: 布尔mask (train_mask, val_mask, or test_mask)
            
        Returns:
            loss, accuracy, f1_score
        """
        self.model.eval()
        
        # 前向传播
        out = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
        )
        
        # 计算损失
        loss = self.criterion(
            out[mask].squeeze(),
            self.data.y[mask].float()
        )
        
        # 预测
        probs = torch.sigmoid(out[mask]).squeeze()
        preds = (probs > 0.5).long()
        labels = self.data.y[mask]
        
        # 准确率
        acc = (preds == labels).float().mean().item()
        
        # F1-Score
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return loss.item(), acc, f1.item()
    
    @torch.no_grad()
    def predict(self, mask, threshold: float = 0.5):
        """
        获取指定mask上的预测结果
        
        Args:
            mask: PyG布尔mask (train/val/test)
            threshold: 二分类阈值（用于返回preds）
        
        Returns:
            probs: 预测概率 (np.ndarray)
            preds: 预测标签 (np.ndarray)
            labels: 真实标签 (np.ndarray)
        """
        self.model.eval()
        
        mask = mask.to(self.device)
        
        out = self.model(
            self.data.x,
            self.data.edge_index,
            self.data.edge_attr if hasattr(self.data, 'edge_attr') else None
        )
        
        logits = out[mask].view(-1)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        labels = self.data.y[mask].view(-1)
        
        return (
            probs.cpu().numpy(),
            preds.cpu().numpy(),
            labels.cpu().numpy()
        )
    
    def train(
        self,
        epochs=200,
        early_stopping_patience=20,
        verbose=True,
        save_dir='outputs/models'
    ):
        """
        完整训练流程
        
        Args:
            epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            verbose: 是否显示详细信息
            save_dir: 模型保存目录
            
        Returns:
            history: 训练历史
        """
        if verbose:
            print("\n" + "="*70)
            print("🚀 GCN训练开始")
            print("="*70)
            self._print_config(epochs, early_stopping_patience)
        
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 早停计数器
        patience_counter = 0
        start_time = time.time()
        
        # 训练循环
        if verbose:
            print("\n" + "="*70)
            print("📈 训练进度")
            print("="*70)
            pbar = tqdm(range(epochs), desc="Training")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()
            
            # 验证集评估
            val_loss, val_acc, val_f1 = self.evaluate(self.data.val_mask)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            
            # 更新进度条
            if verbose:
                pbar.set_postfix({
                    'T_Loss': f'{train_loss:.4f}',
                    'V_Loss': f'{val_loss:.4f}',
                    'V_F1': f'{val_f1:.4f}',
                    'LR': f'{current_lr:.6f}'
                })
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"\n   ⭐ Epoch {epoch+1}: 新的最佳模型! Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\n   🛑 Early stopping triggered at epoch {epoch+1}")
                    print(f"   📌 Best epoch: {self.best_epoch+1}")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        
        if verbose:
            print("\n" + "="*70)
            print("✅ 训练完成")
            print("="*70)
            print(f"\n   总训练时间: {total_time:.2f}s")
            print(f"   最佳Epoch: {self.best_epoch + 1}")
            print(f"   最佳Val Loss: {self.best_val_loss:.4f}")
            print(f"   最佳Val F1: {max(self.history['val_f1']):.4f}")
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # 保存最佳模型
        self.save_checkpoint(save_dir / 'best_model.pt')
        
        # 保存训练历史
        self._save_history(save_dir / 'training_history.json')
        
        if verbose:
            print(f"\n   💾 模型已保存: {save_dir / 'best_model.pt'}")
            print(f"   📊 训练历史已保存: {save_dir / 'training_history.json'}")
        
        return self.history
    
    def save_checkpoint(self, path):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        return checkpoint
    
    def _print_config(self, epochs, patience):
        """打印训练配置"""
        print("\n配置:")
        print(f"   模型: GCN")
        print(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   设备: {self.device}")
        print(f"   优化器: Adam")
        print(f"   初始学习率: {self.optimizer.param_groups[0]['lr']}")
        print(f"   权重衰减: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"   最大Epochs: {epochs}")
        print(f"   早停耐心: {patience}")
        
        # 数据集信息
        print(f"\n数据集:")
        print(f"   训练集: {self.data.train_mask.sum().item()} 节点")
        print(f"   验证集: {self.data.val_mask.sum().item()} 节点")
        print(f"   测试集: {self.data.test_mask.sum().item()} 节点")
        print(f"   训练集离职率: {self.data.y[self.data.train_mask].float().mean().item():.2%}")
    
    def _save_history(self, path):
        """保存训练历史"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def quick_train(
    model,
    data,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    early_stopping_patience=20,
    device='cpu',
    save_dir='outputs/models'
):
    """
    快速训练函数（简化接口）
    
    Args:
        model: GCN模型
        data: PyG Data对象
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        early_stopping_patience: 早停耐心
        device: 设备
        save_dir: 保存目录
        
    Returns:
        trainer: 训练器对象
        history: 训练历史
    """
    # 计算类别权重
    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    pos_weight = num_neg / num_pos  if num_pos > 0 else 1.0
    
    print(f"\n类别平衡:")
    print(f"   正样本(离职): {num_pos}")
    print(f"   负样本(在职): {num_neg}")
    print(f"   正样本权重: {pos_weight:.2f}")
    
    # 创建训练器
    trainer = GCNTrainer(
        model=model,
        data=data,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight=pos_weight
    )
    
    # 训练
    history = trainer.train(
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir
    )
    
    return trainer, history


if __name__ == '__main__':
    """测试训练器"""
    print("\n" + "="*70)
    print("🧪 GCN训练器测试")
    print("="*70)
    
    # 加载数据和模型
    print("\n1. 加载数据...")
    data = torch.load('data/processed/homo_graph.pt')
    print(f"   ✓ 数据加载成功")
    
    print("\n2. 创建模型...")
    from gcn import create_gcn_model
    model = create_gcn_model(
        in_channels=data.num_node_features,
        architecture='default',
        dropout=0.5
    )
    print(f"   ✓ 模型创建成功")
    
    print("\n3. 开始训练...")
    trainer, history = quick_train(
        model=model,
        data=data,
        epochs=50,  # 测试用少量epoch
        lr=0.01,
        early_stopping_patience=10,
        device='cpu'
    )
    
    print("\n4. 测试集评估...")
    test_loss, test_acc, test_f1 = trainer.evaluate(data.test_mask)
    print(f"   测试Loss: {test_loss:.4f}")
    print(f"   测试Acc: {test_acc:.4f}")
    print(f"   测试F1: {test_f1:.4f}")
    
    print("\n" + "="*70)
    print("✅ 训练器测试完成")
    print("="*70)
