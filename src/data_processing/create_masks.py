"""
快速创建mask文件
"""
import numpy as np
import torch
from pathlib import Path

# 加载索引
train_idx = np.load('/Users/yu/code/code2510/gnn/data/splits/train_idx.npy')
val_idx = np.load('/Users/yu/code/code2510/gnn/data/splits/val_idx.npy')
test_idx = np.load('/Users/yu/code/code2510/gnn/data/splits/test_idx.npy')

# 创建mask
n_samples = 500

train_mask = torch.zeros(n_samples, dtype=torch.bool)
val_mask = torch.zeros(n_samples, dtype=torch.bool)
test_mask = torch.zeros(n_samples, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# 保存
output_dir = Path('/Users/yu/code/code2510/gnn/data/splits')
torch.save(train_mask, output_dir / 'train_mask.pt')
torch.save(val_mask, output_dir / 'val_mask.pt')
torch.save(test_mask, output_dir / 'test_mask.pt')

print("🎭 Mask创建完成！")
print(f"   ✅ train_mask.pt: {train_mask.sum()} True")
print(f"   ✅ val_mask.pt: {val_mask.sum()} True")  
print(f"   ✅ test_mask.pt: {test_mask.sum()} True")
print(f"\n✅ 已保存到: {output_dir}")