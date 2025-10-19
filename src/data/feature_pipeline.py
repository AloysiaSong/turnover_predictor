"""
特征工程管道
============
功能：
1. Top-N连续特征选择（基于AUPR/MI/LR系数）
2. 分箱 + One-hot编码
3. 二阶交互项
4. 严格防止数据泄漏（仅在Train上fit）
"""

import torch
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class NodeFeatureTransformer:
    """
    节点特征转换器
    
    支持功能：
    1. 连续特征重要性评估（AUPR/MI/LR系数）
    2. Top-N特征分箱 + One-hot
    3. 二阶交互项
    
    重要：所有统计量仅在训练集上计算！
    """
    
    def __init__(
        self,
        topn: int = 3,
        n_bins: int = 5,
        add_interactions: bool = True,
        importance_metric: str = 'aupr',
        random_state: int = 42
    ):
        """
        参数:
            topn: 选择前N个重要特征
            n_bins: 分箱数量
            add_interactions: 是否添加交互项
            importance_metric: 重要性指标 {'aupr', 'mi', 'lr_coef'}
            random_state: 随机种子
        """
        self.topn = topn
        self.n_bins = n_bins
        self.add_interactions = add_interactions
        self.importance_metric = importance_metric
        self.random_state = random_state
        
        # 状态（fit后保存）
        self.is_fitted = False
        self.top_feature_indices: Optional[List[int]] = None
        self.feature_names: Optional[List[str]] = None
        self.bin_edges: Optional[Dict[int, np.ndarray]] = None
        self.scaler: Optional[StandardScaler] = None
        self.n_original_features: Optional[int] = None
        self.n_augmented_features: Optional[int] = None
        
    def _compute_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        计算特征重要性
        
        参数:
            X: [N, D] 特征矩阵
            y: [N] 标签
            
        返回:
            importance: [D] 重要性分数
        """
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        if self.importance_metric == 'aupr':
            # 使用AUPR作为重要性
            for i in range(n_features):
                try:
                    # 简单的单特征预测能力
                    feature = X[:, i].reshape(-1, 1)
                    lr = LogisticRegression(random_state=self.random_state, max_iter=100)
                    lr.fit(feature, y)
                    y_prob = lr.predict_proba(feature)[:, 1]
                    importance[i] = average_precision_score(y, y_prob)
                except:
                    importance[i] = 0.0
                    
        elif self.importance_metric == 'mi':
            # 互信息
            importance = mutual_info_classif(
                X, y,
                random_state=self.random_state,
                n_neighbors=min(3, len(y) // 10)
            )
            
        elif self.importance_metric == 'lr_coef':
            # 逻辑回归系数绝对值
            lr = LogisticRegression(
                random_state=self.random_state,
                max_iter=200,
                penalty='l2',
                C=1.0
            )
            lr.fit(X, y)
            importance = np.abs(lr.coef_[0])
            
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")
        
        return importance
    
    def _identify_continuous_features(self, X: np.ndarray) -> List[int]:
        """
        识别连续特征（非二值）
        
        参数:
            X: [N, D] 特征矩阵
            
        返回:
            continuous_indices: 连续特征索引列表
        """
        continuous_indices = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            unique_values = np.unique(X[:, i])
            # 如果unique值数量>10，认为是连续特征
            if len(unique_values) > 10:
                continuous_indices.append(i)
        
        return continuous_indices
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor
    ) -> 'NodeFeatureTransformer':
        """
        拟合转换器（仅使用训练集）
        
        参数:
            X: [N, D] 全部节点特征
            y: [N] 全部标签
            train_mask: [N] 训练集mask
            
        返回:
            self
        """
        print("\n" + "="*70)
        print("🔧 特征工程管道拟合")
        print("="*70)
        
        # 转换为numpy
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        train_mask_np = train_mask.cpu().numpy()
        
        # 仅使用训练集
        X_train = X_np[train_mask_np]
        y_train = y_np[train_mask_np]
        
        self.n_original_features = X_np.shape[1]
        
        print(f"\n1. 训练集统计:")
        print(f"   样本数: {len(X_train)}")
        print(f"   正样本: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"   特征数: {self.n_original_features}")
        
        # 识别连续特征
        continuous_indices = self._identify_continuous_features(X_train)
        print(f"\n2. 识别连续特征: {len(continuous_indices)}个")
        
        if len(continuous_indices) == 0:
            print("   ⚠️  未找到连续特征，跳过特征扩展")
            self.is_fitted = True
            self.n_augmented_features = self.n_original_features
            return self
        
        # 计算重要性（仅在连续特征上）
        print(f"\n3. 计算特征重要性 (metric={self.importance_metric})...")
        X_continuous = X_train[:, continuous_indices]
        importance = self._compute_feature_importance(X_continuous, y_train)
        
        # 选择Top-N
        topn = min(self.topn, len(continuous_indices))
        top_indices_local = np.argsort(importance)[-topn:][::-1]
        self.top_feature_indices = [continuous_indices[i] for i in top_indices_local]
        
        print(f"\n4. Top-{topn} 重要特征:")
        for rank, idx in enumerate(self.top_feature_indices, 1):
            imp = importance[top_indices_local[rank-1]]
            print(f"   #{rank}: 特征{idx} (重要性={imp:.4f})")
        
        # 分箱（仅在训练集上计算分位点）
        print(f"\n5. 分箱 (n_bins={self.n_bins})...")
        self.bin_edges = {}
        
        for idx in self.top_feature_indices:
            feature_train = X_train[:, idx]
            # 计算分位点
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(feature_train, quantiles)
            # 确保边界唯一
            edges = np.unique(edges)
            self.bin_edges[idx] = edges
            print(f"   特征{idx}: {len(edges)-1}个箱 {edges[:3]}...{edges[-3:]}")
        
        # 标准化器（用于交互项）
        if self.add_interactions and topn >= 2:
            print(f"\n6. 标准化器拟合（用于交互项）...")
            X_top_train = X_train[:, self.top_feature_indices]
            self.scaler = StandardScaler()
            self.scaler.fit(X_top_train)
        
        self.is_fitted = True
        
        # 计算扩展后的特征数
        n_onehot = sum(len(self.bin_edges[idx]) - 1 for idx in self.top_feature_indices)
        n_interactions = topn * (topn - 1) // 2 if self.add_interactions and topn >= 2 else 0
        self.n_augmented_features = self.n_original_features + n_onehot + n_interactions
        
        print(f"\n7. 特征扩展完成:")
        print(f"   原始特征: {self.n_original_features}")
        print(f"   分箱特征: {n_onehot}")
        print(f"   交互特征: {n_interactions}")
        print(f"   总计: {self.n_augmented_features} (+{self.n_augmented_features - self.n_original_features})")
        
        print("\n" + "="*70)
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        转换特征
        
        参数:
            X: [N, D] 节点特征
            
        返回:
            X_aug: [N, D'] 扩展后的特征
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        
        if self.top_feature_indices is None:
            # 没有连续特征，直接返回
            return X
        
        X_np = X.cpu().numpy()
        
        # 1. 分箱 + One-hot
        onehot_features = []
        for idx in self.top_feature_indices:
            feature = X_np[:, idx]
            edges = self.bin_edges[idx]
            
            # 数字化
            bins = np.digitize(feature, edges[1:-1])  # 不包括最小和最大边界
            bins = np.clip(bins, 0, len(edges) - 2)  # 确保在范围内
            
            # One-hot
            n_bins = len(edges) - 1
            onehot = np.zeros((len(feature), n_bins))
            onehot[np.arange(len(feature)), bins] = 1
            
            onehot_features.append(onehot)
        
        onehot_features = np.hstack(onehot_features) if onehot_features else np.empty((len(X_np), 0))
        
        # 2. 交互项
        interaction_features = np.empty((len(X_np), 0))
        if self.add_interactions and len(self.top_feature_indices) >= 2:
            X_top = X_np[:, self.top_feature_indices]
            X_top_scaled = self.scaler.transform(X_top)
            
            # 两两相乘
            interactions = []
            for i in range(len(self.top_feature_indices)):
                for j in range(i + 1, len(self.top_feature_indices)):
                    interaction = X_top_scaled[:, i] * X_top_scaled[:, j]
                    interactions.append(interaction.reshape(-1, 1))
            
            if interactions:
                interaction_features = np.hstack(interactions)
        
        # 3. 拼接
        X_aug = np.hstack([X_np, onehot_features, interaction_features])
        
        return torch.from_numpy(X_aug).float().to(X.device)
    
    def fit_transform(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        拟合并转换
        
        参数:
            X: [N, D] 节点特征
            y: [N] 标签
            train_mask: [N] 训练集mask
            
        返回:
            X_aug: [N, D'] 扩展后的特征
        """
        self.fit(X, y, train_mask)
        return self.transform(X)
    
    def save(self, path: str):
        """保存转换器状态"""
        state = {
            'topn': self.topn,
            'n_bins': self.n_bins,
            'add_interactions': self.add_interactions,
            'importance_metric': self.importance_metric,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'top_feature_indices': self.top_feature_indices,
            'feature_names': self.feature_names,
            'bin_edges': self.bin_edges,
            'scaler': self.scaler,
            'n_original_features': self.n_original_features,
            'n_augmented_features': self.n_augmented_features
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✓ 特征转换器已保存: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NodeFeatureTransformer':
        """加载转换器状态"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        transformer = cls(
            topn=state['topn'],
            n_bins=state['n_bins'],
            add_interactions=state['add_interactions'],
            importance_metric=state['importance_metric'],
            random_state=state['random_state']
        )
        
        transformer.is_fitted = state['is_fitted']
        transformer.top_feature_indices = state['top_feature_indices']
        transformer.feature_names = state['feature_names']
        transformer.bin_edges = state['bin_edges']
        transformer.scaler = state['scaler']
        transformer.n_original_features = state['n_original_features']
        transformer.n_augmented_features = state['n_augmented_features']
        
        print(f"✓ 特征转换器已加载: {path}")
        
        return transformer
    
    def get_augmented_feature_dim(self) -> int:
        """获取扩展后的特征维度"""
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted")
        return self.n_augmented_features
