"""
MLP基线模型
用于离职预测的二分类任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """
    多层感知机基线模型
    """
    
    def __init__(self,
                 input_dim: int = 47,
                 hidden_dims: list = [128, 64, 32],
                 dropout: float = 0.5):
        """
        Args:
            input_dim: 输入特征维度 (默认47)
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比例
        """
        super(MLPBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            logits: (batch_size, 1)
        """
        return self.network(x)
    
    def predict_proba(self, x):
        """
        预测概率
        
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            probs: (batch_size, 1) 范围[0, 1]
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x, threshold=0.5):
        """
        预测类别
        
        Args:
            x: (batch_size, input_dim)
            threshold: 分类阈值
            
        Returns:
            preds: (batch_size, 1) 0或1
        """
        probs = self.predict_proba(x)
        preds = (probs >= threshold).long()
        return preds


def create_mlp_model(input_dim: int = 47, 
                     architecture: str = 'default',
                     dropout: float = 0.5):
    """
    创建不同架构的MLP模型
    
    Args:
        input_dim: 输入维度
        architecture: 架构类型 ('shallow', 'default', 'deep', 'very_deep')
        dropout: Dropout比例
        
    Returns:
        model: MLPBaseline实例
    """
    architectures = {
        'shallow': [64, 32],
        'default': [128, 64, 32],
        'deep': [256, 128, 64, 32],
        'very_deep': [512, 256, 128, 64, 32]
    }
    
    if architecture not in architectures:
        raise ValueError(f"未知架构: {architecture}. 可选: {list(architectures.keys())}")
    
    hidden_dims = architectures[architecture]
    
    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )
    
    # 打印模型信息
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"MLP模型创建成功")
    print(f"{'='*60}")
    print(f"架构: {architecture}")
    print(f"层结构: {input_dim} → {' → '.join(map(str, hidden_dims))} → 1")
    print(f"总参数量: {n_params:,}")
    print(f"Dropout: {dropout}")
    
    return model


def test_model():
    """测试模型"""
    print("="*60)
    print("MLP模型测试")
    print("="*60)
    
    # 创建模型
    model = create_mlp_model(input_dim=47, architecture='default')
    
    print(f"\n模型结构:")
    print(model)
    
    # 测试前向传播
    batch_size = 16
    x = torch.randn(batch_size, 47)
    
    print(f"\n{'='*60}")
    print(f"前向传播测试")
    print(f"{'='*60}")
    print(f"输入形状: {x.shape}")
    
    # Logits
    logits = model(x)
    print(f"Logits形状: {logits.shape}")
    print(f"Logits范围: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # 概率
    probs = model.predict_proba(x)
    print(f"\n概率形状: {probs.shape}")
    print(f"概率范围: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"平均概率: {probs.mean():.4f}")
    
    # 预测
    preds = model.predict(x, threshold=0.5)
    print(f"\n预测形状: {preds.shape}")
    print(f"预测值: {preds.squeeze()}")
    print(f"正样本数: {preds.sum().item()}/{batch_size}")
    
    # 参数统计
    print(f"\n{'='*60}")
    print(f"参数统计")
    print(f"{'='*60}")
    for name, param in model.named_parameters():
        print(f"{name:30s} {str(param.shape):20s} {param.numel():>10,} 参数")
    
    print("\n✅ 模型测试通过！")


if __name__ == '__main__':
    test_model()