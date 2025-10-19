"""
GCN模型实现（增强版）
====================
新增功能:
1. EdgeDropout (DropEdge) - 训练时随机丢弃边
2. FeatureDropout - 输入特征dropout
3. 灵活配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


def drop_edges(
    edge_index: torch.Tensor,
    p: float,
    num_nodes: int,
    training: bool = True
) -> torch.Tensor:
    """
    EdgeDropout (DropEdge)
    
    在训练时随机丢弃边，防止过拟合，提高泛化能力
    
    参数:
        edge_index: [2, E] 边索引
        p: dropout概率 (0.0-1.0)
        num_nodes: 节点数量
        training: 是否训练模式
        
    返回:
        edge_index: 丢弃边后的边索引
        
    参考:
        DropEdge: Sampling Edge-Dropout for Graph Neural Networks (ICLR 2020)
    """
    if not training or p == 0.0:
        return edge_index
    
    # 生成保留mask
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > p
    
    # 应用mask
    edge_index = edge_index[:, keep_mask]
    
    return edge_index


class GCN(nn.Module):
    """
    图卷积网络（GCN）
    
    参数:
        in_channels: 输入特征维度
        hidden_channels: 隐藏层维度
        out_channels: 输出维度（通常=1，二分类）
        num_layers: GCN层数
        dropout: 标准dropout率
        edge_dropout: 边dropout率 (DropEdge)
        feature_dropout: 特征dropout率
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5,
        edge_dropout: float = 0.0,
        feature_dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.feature_dropout = feature_dropout
        
        # Feature dropout层
        self.feat_drop = nn.Dropout(feature_dropout) if feature_dropout > 0 else nn.Identity()
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            # 单层GCN直接输出
            self.convs = nn.ModuleList([GCNConv(in_channels, out_channels)])
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: 边特征（当前未使用）
            
        返回:
            out: [N, 1] 输出logits
        """
        num_nodes = x.size(0)
        
        # 1. Feature Dropout（训练时）
        x = self.feat_drop(x)
        
        # 2. Edge Dropout（训练时）
        if self.training and self.edge_dropout > 0:
            edge_index = drop_edges(
                edge_index,
                p=self.edge_dropout,
                num_nodes=num_nodes,
                training=True
            )
        
        # 3. GCN层
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_config(self) -> dict:
        """返回模型配置"""
        return {
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'edge_dropout': self.edge_dropout,
            'feature_dropout': self.feature_dropout
        }


def create_gcn_model(
    in_channels: int,
    architecture: str = 'default',
    dropout: float = 0.5,
    edge_dropout: float = 0.0,
    feature_dropout: float = 0.0
) -> GCN:
    """
    创建GCN模型
    
    参数:
        in_channels: 输入特征维度
        architecture: 架构类型 {'shallow', 'default', 'deep', 'very_deep'}
        dropout: 标准dropout
        edge_dropout: 边dropout
        feature_dropout: 特征dropout
        
    返回:
        model: GCN模型
    """
    arch_configs = {
        'shallow': {'num_layers': 1, 'hidden_channels': 32},
        'default': {'num_layers': 2, 'hidden_channels': 64},
        'deep': {'num_layers': 3, 'hidden_channels': 128},
        'very_deep': {'num_layers': 4, 'hidden_channels': 256}
    }
    
    if architecture not in arch_configs:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    config = arch_configs[architecture]
    
    model = GCN(
        in_channels=in_channels,
        hidden_channels=config['hidden_channels'],
        out_channels=1,
        num_layers=config['num_layers'],
        dropout=dropout,
        edge_dropout=edge_dropout,
        feature_dropout=feature_dropout
    )
    
    return model


# 预定义架构配置
ARCHITECTURES = {
    'shallow': {
        'num_layers': 1,
        'hidden_channels': 32,
        'description': 'Single GCN layer, lightweight'
    },
    'default': {
        'num_layers': 2,
        'hidden_channels': 64,
        'description': 'Standard 2-layer GCN'
    },
    'deep': {
        'num_layers': 3,
        'hidden_channels': 128,
        'description': 'Deeper 3-layer GCN'
    },
    'very_deep': {
        'num_layers': 4,
        'hidden_channels': 256,
        'description': 'Very deep 4-layer GCN'
    }
}