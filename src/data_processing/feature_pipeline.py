"""
特征工程管道
================
实现Top-N连续特征的非线性扩展：
1. 特征重要性评估
2. 分箱one-hot编码
3. 二阶交互项
4. 严格的Train/Val/Test分离
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import mutual_info_classif
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class NodeFeatureTransformer:
    """
    节点特征转换器
    
    功能：
    1. 选择Top-N重要连续特征
    2. 分箱并one-hot编码
    3. 生成二阶交互项
    4. 仅使用Train数据拟合，避免数据泄漏
    
    参数:
        topn: 选择的Top特征数量
        n_bins: 分箱数量
        add_interactions: 是否添加二阶交互
        importance_metric: 特征重要性评估方法 {'aupr', 'mi', 'lr_coef'}
    """
    
    def __init__(
        self,
        topn: int = 3,
        n_bins: int = 5,
        add_interactions: bool = True,
        importance_metric: str = 'aupr'
    ):
        self.topn = topn
        self.n_bins = n_bins
        self.add_interactions = add_interactions
        self.importance_metric = importance_metric
        
        # 状态变量（仅从Train拟合）
        self.selected_indices: Optional[List[int]] = None
        self.feature_scores: Optional[np.ndarray] = None
        self.bin_edges: Optional[Dict[int, np.ndarray]] = None
        self.scalers: Optional[Dict[int, StandardScaler]] = None
        self.is_fitted: bool = False
        
        # 元数据
        self.original_dim: Optional[int] = None
        self.augmented_dim: Optional[int] = None
        
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> 'NodeFeatureTransformer':
        """
        在训练集上拟合转换器
        
        参数:
            x_train: 训练集特征 [N_train, D]
            y_train: 训练集标签 [N_train]
            feature_names: 特征名列表（可选）
        """
        x_np = x_train.cpu().numpy()
        y_np = y_train.cpu().numpy()
        
        self.original_dim = x_np.shape[1]
        
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(self.original_dim)]
        self.feature_names = feature_names
        
        print(f"\n{'='*70}")
        print(f"特征工程管道：拟合")
        print(f"{'='*70}")
        print(f"训练样本数: {len(x_np)}")
        print(f"原始特征数: {self.original_dim}")
        print(f"重要性指标: {self.importance_metric}")
        
        # 1. 评估特征重要性
        self.feature_scores = self._compute_feature_importance(x_np, y_np)
        
        # 2. 选择Top-N特征
        self.selected_indices = np.argsort(self.feature_scores)[-self.topn:].tolist()
        
        print(f"\nTop-{self.topn} 重要特征:")
        for i, idx in enumerate(self.selected_indices, 1):
            print(f"   #{i}: {feature_names[idx]:<20} (score={self.feature_scores[idx]:.4f})")
        
        # 3. 为每个选定特征计算分箱边界（仅使用Train）
        self.bin_edges = {}
        self.scalers = {}
        
        for idx in self.selected_indices:
            feat_values = x_np[:, idx]
            
            # 计算分位点
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(feat_values, quantiles)
            # 确保边界唯一
            edges = np.unique(edges)
            self.bin_edges[idx] = edges
            
            # 标准化器（用于交互项）
            scaler = StandardScaler()
            scaler.fit(feat_values.reshape(-1, 1))
            self.scalers[idx] = scaler
            
            print(f"\n特征 {feature_names[idx]}:")
            print(f"   分箱边界: {edges}")
            print(f"   均值/标准差: {scaler.mean_[0]:.4f} / {scaler.scale_[0]:.4f}")
        
        self.is_fitted = True
        
        # 计算增强后维度
        binned_dim = sum(len(self.bin_edges[idx]) - 1 for idx in self.selected_indices)
        interaction_dim = 0
        if self.add_interactions:
            interaction_dim = len(self.selected_indices) * (len(self.selected_indices) - 1) // 2
        
        self.augmented_dim = self.original_dim + binned_dim + interaction_dim
        
        print(f"\n增强维度:")
        print(f"   原始特征: {self.original_dim}")
        print(f"   分箱one-hot: {binned_dim}")
        print(f"   交互项: {interaction_dim}")
        print(f"   总计: {self.augmented_dim}")
        
        return self
    
    def _compute_feature_importance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算每个特征的重要性
        
        返回: [D] 重要性分数
        """
        n_features = x.shape[1]
        scores = np.zeros(n_features)
        
        if self.importance_metric == 'aupr':
            # 单变量逻辑回归AUPR
            print(f"计算特征重要性 (AUPR)...")
            for i in range(n_features):
                try:
                    lr = LogisticRegression(max_iter=100, random_state=42)
                    lr.fit(x[:, i:i+1], y)
                    y_prob = lr.predict_proba(x[:, i:i+1])[:, 1]
                    scores[i] = average_precision_score(y, y_prob)
                except:
                    scores[i] = 0.0
                    
        elif self.importance_metric == 'mi':
            # 互信息
            print(f"计算特征重要性 (互信息)...")
            scores = mutual_info_classif(x, y, random_state=42)
            
        elif self.importance_metric == 'lr_coef':
            # 多变量逻辑回归系数绝对值
            print(f"计算特征重要性 (LR系数)...")
            lr = LogisticRegression(max_iter=200, random_state=42)
            lr.fit(x, y)
            scores = np.abs(lr.coef_[0])
        
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")
        
        return scores
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        转换特征
        
        参数:
            x: 输入特征 [N, D]
            
        返回:
            x_aug: 增强特征 [N, D_aug]
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        
        x_np = x.cpu().numpy()
        device = x.device
        dtype = x.dtype
        
        augmented_features = [x_np]  # 保留原始特征
        
        # 1. 分箱one-hot
        for idx in self.selected_indices:
            feat_values = x_np[:, idx]
            edges = self.bin_edges[idx]
            
            # 数字化（返回bin索引）
            bins = np.digitize(feat_values, edges[1:-1])  # 不包括首尾
            
            # one-hot编码
            n_bins_actual = len(edges) - 1
            one_hot = np.zeros((len(feat_values), n_bins_actual))
            one_hot[np.arange(len(feat_values)), bins] = 1
            
            augmented_features.append(one_hot)
        
        # 2. 二阶交互项
        if self.add_interactions and len(self.selected_indices) > 1:
            for i in range(len(self.selected_indices)):
                for j in range(i + 1, len(self.selected_indices)):
                    idx_i = self.selected_indices[i]
                    idx_j = self.selected_indices[j]
                    
                    # 标准化后相乘
                    feat_i = self.scalers[idx_i].transform(x_np[:, idx_i:idx_i+1]).flatten()
                    feat_j = self.scalers[idx_j].transform(x_np[:, idx_j:idx_j+1]).flatten()
                    
                    interaction = (feat_i * feat_j).reshape(-1, 1)
                    augmented_features.append(interaction)
        
        # 拼接所有特征
        x_aug_np = np.concatenate(augmented_features, axis=1)
        
        # 转回Tensor
        x_aug = torch.from_numpy(x_aug_np).to(device).to(dtype)
        
        return x_aug
    
    def save(self, path: str):
        """保存转换器状态"""
        state = {
            'topn': self.topn,
            'n_bins': self.n_bins,
            'add_interactions': self.add_interactions,
            'importance_metric': self.importance_metric,
            'selected_indices': self.selected_indices,
            'feature_scores': self.feature_scores,
            'bin_edges': self.bin_edges,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'original_dim': self.original_dim,
            'augmented_dim': self.augmented_dim,
            'is_fitted': self.is_fitted
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        # 同时保存可读的JSON
        json_path = path.replace('.pkl', '_summary.json')
        json_state = {
            'config': {
                'topn': self.topn,
                'n_bins': self.n_bins,
                'add_interactions': self.add_interactions,
                'importance_metric': self.importance_metric
            },
            'selected_features': [
                {
                    'index': idx,
                    'name': self.feature_names[idx],
                    'score': float(self.feature_scores[idx]),
                    'bin_edges': self.bin_edges[idx].tolist()
                }
                for idx in self.selected_indices
            ],
            'dimensions': {
                'original': self.original_dim,
                'augmented': self.augmented_dim
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_state, f, indent=2)
        
        print(f"\n✓ 转换器已保存:")
        print(f"   State: {path}")
        print(f"   Summary: {json_path}")
    
    @classmethod
    def load(cls, path: str) -> 'NodeFeatureTransformer':
        """加载转换器状态"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        transformer = cls(
            topn=state['topn'],
            n_bins=state['n_bins'],
            add_interactions=state['add_interactions'],
            importance_metric=state['importance_metric']
        )
        
        transformer.selected_indices = state['selected_indices']
        transformer.feature_scores = state['feature_scores']
        transformer.bin_edges = state['bin_edges']
        transformer.scalers = state['scalers']
        transformer.feature_names = state['feature_names']
        transformer.original_dim = state['original_dim']
        transformer.augmented_dim = state['augmented_dim']
        transformer.is_fitted = state['is_fitted']
        
        print(f"✓ 转换器已加载: {path}")
        
        return transformer


def apply_feature_augmentation(
    data,
    train_mask: torch.Tensor,
    config: Dict,
    save_path: Optional[str] = None
) -> Tuple[torch.Tensor, Optional[NodeFeatureTransformer]]:
    """
    应用特征增强到图数据
    
    参数:
        data: PyG Data对象
        train_mask: 训练集mask
        config: 配置字典
        save_path: 保存路径
        
    返回:
        x_aug: 增强后的特征
        transformer: 转换器对象（若启用）
    """
    if not config.get('feat_augment', False):
        return data.x, None
    
    # 创建并拟合转换器
    transformer = NodeFeatureTransformer(
        topn=config.get('topn', 3),
        n_bins=config.get('bins', 5),
        add_interactions=config.get('add_interactions', True),
        importance_metric=config.get('importance', 'aupr')
    )
    
    # 仅在训练集上拟合
    x_train = data.x[train_mask]
    y_train = data.y[train_mask]
    
    transformer.fit(x_train, y_train)
    
    # 转换所有数据
    x_aug = transformer.transform(data.x)
    
    # 保存
    if save_path:
        transformer.save(save_path)
    
    return x_aug, transformer
