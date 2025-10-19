"""
数据集划分模块
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict
import json
from pathlib import Path


class DataSplitter:
    """数据集划分器（支持分层抽样）"""
    
    def __init__(self, 
                 train_ratio: float = 0.68,
                 val_ratio: float = 0.12,
                 test_ratio: float = 0.20,
                 random_state: int = 42):
        """
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            random_state: 随机种子
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "比例之和必须为1"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
    
    def split(self, y: np.ndarray, n_samples: int) -> Dict[str, np.ndarray]:
        """
        分层划分数据集
        
        Args:
            y: 标签数组 (n_samples,)
            n_samples: 总样本数
            
        Returns:
            {
                'train_idx': 训练集索引,
                'val_idx': 验证集索引,
                'test_idx': 测试集索引
            }
        """
        print("\n" + "="*60)
        print("数据集划分 (分层抽样)")
        print("="*60)
        
        # 第一次划分: train+val vs test
        print(f"\n🔹 第一次划分: train+val ({self.train_ratio+self.val_ratio:.0%}) vs test ({self.test_ratio:.0%})")
        
        splitter1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_ratio,
            random_state=self.random_state
        )
        
        train_val_idx, test_idx = next(splitter1.split(range(n_samples), y))
        
        print(f"   Train+Val: {len(train_val_idx)} 样本")
        print(f"   Test: {len(test_idx)} 样本")
        
        # 第二次划分: train vs val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        print(f"\n🔹 第二次划分: train ({self.train_ratio:.0%}) vs val ({self.val_ratio:.0%})")
        
        splitter2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio_adjusted,
            random_state=self.random_state
        )
        
        y_train_val = y[train_val_idx]
        train_idx_local, val_idx_local = next(splitter2.split(
            range(len(train_val_idx)), y_train_val
        ))
        
        # 映射回全局索引
        train_idx = train_val_idx[train_idx_local]
        val_idx = train_val_idx[val_idx_local]
        
        print(f"   Train: {len(train_idx)} 样本")
        print(f"   Val: {len(val_idx)} 样本")
        
        # 验证分布
        print("\n" + "="*60)
        print("标签分布验证")
        print("="*60)
        
        self._print_label_distribution(y, train_idx, val_idx, test_idx)
        
        # 保存索引
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        
        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
    
    def _print_label_distribution(self, y, train_idx, val_idx, test_idx):
        """打印标签分布"""
        sets = {
            'Train': train_idx,
            'Val': val_idx,
            'Test': test_idx,
            'Total': np.arange(len(y))
        }
        
        print(f"{'集合':<10} {'样本数':<10} {'正样本':<10} {'负样本':<10} {'正样本率':<10}")
        print("-" * 60)
        
        for name, idx in sets.items():
            y_subset = y[idx]
            pos = y_subset.sum()
            neg = len(y_subset) - pos
            pos_rate = pos / len(y_subset) * 100
            
            print(f"{name:<10} {len(y_subset):<10} {int(pos):<10} {int(neg):<10} {pos_rate:<10.1f}%")
    
    def save(self, output_dir: str):
        """保存划分结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        np.save(output_dir / 'train_idx.npy', self.train_idx)
        np.save(output_dir / 'val_idx.npy', self.val_idx)
        np.save(output_dir / 'test_idx.npy', self.test_idx)
        
        # 保存配置
        config = {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_state': self.random_state,
            'train_size': int(len(self.train_idx)),
            'val_size': int(len(self.val_idx)),
            'test_size': int(len(self.test_idx))
        }
        
        with open(output_dir / 'split_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ 划分结果已保存到: {output_dir}")

# 在 data_splitter.py 中添加这个方法

def create_masks(self, n_samples: int) -> Dict[str, torch.BoolTensor]:
    """
    创建PyG风格的mask
    
    Args:
        n_samples: 总样本数
        
    Returns:
        {
            'train_mask': 训练mask,
            'val_mask': 验证mask,
            'test_mask': 测试mask
        }
    """
    import torch
    
    if self.train_idx is None:
        raise ValueError("请先调用split()方法进行数据划分")
    
    print("\n🎭 创建mask...")
    
    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)
    
    train_mask[self.train_idx] = True
    val_mask[self.val_idx] = True
    test_mask[self.test_idx] = True
    
    print(f"   ✅ Train mask: {train_mask.sum()} True")
    print(f"   ✅ Val mask: {val_mask.sum()} True")
    print(f"   ✅ Test mask: {test_mask.sum()} True")
    
    return {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }

def main():
    """演示数据划分"""
    import sys
    sys.path.append('/Users/yu/code/code2510/gnn')
    
    # 加载标签
    y_binary = np.load('/Users/yu/code/code2510/gnn/data/processed/y_turnover_binary.npy')
    
    # 划分数据
    splitter = DataSplitter(
        train_ratio=0.68,
        val_ratio=0.12,
        test_ratio=0.20,
        random_state=42
    )
    
    split_dict = splitter.split(y_binary, len(y_binary))
    
    # 保存
    splitter.save('/Users/yu/code/code2510/gnn/data/splits')
    
    print("\n✅ 数据划分完成！")
    print("\n📊 划分结果:")
    print(f"   训练集: {len(split_dict['train_idx'])} 样本")
    print(f"   验证集: {len(split_dict['val_idx'])} 样本")
    print(f"   测试集: {len(split_dict['test_idx'])} 样本")


if __name__ == '__main__':
    main()