"""
同构图构建器
将异构图转换为同构图（所有节点视为同一类型）

核心思路:
1. 只使用员工节点（500个）
2. 基于员工之间的相似性构建边
3. 多种边构建策略：共同属性、技能相似度、k-NN
4. 确保图连通性
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
import json


class HomogeneousGraphBuilder:
    """同构图构建器"""
    
    def __init__(self, data_dir='data'):
        """
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.edges_dir = self.data_dir / 'edges'
        self.splits_dir = self.data_dir / 'splits'
        
        # 统计信息
        self.stats = {}
        
    def build(self, strategy='hybrid', k=10, similarity_threshold=0.5):
        """
        构建同构图
        
        Args:
            strategy: 边构建策略
                - 'attribute': 基于共同属性
                - 'similarity': 基于特征相似度
                - 'knn': 基于k近邻
                - 'hybrid': 混合策略（推荐）
            k: k-NN中的k值
            similarity_threshold: 相似度阈值
            
        Returns:
            data: PyG Data对象
        """
        print("\n" + "="*70)
        print("🔨 同构图构建器")
        print("="*70)
        
        # 1. 加载数据
        print("\n📂 Step 1/6: 加载数据...")
        X, y, df = self._load_data()
        
        # 2. 构建边
        print(f"\n🔗 Step 2/6: 构建边 (策略={strategy})...")
        edge_index, edge_weights = self._build_edges(df, X, strategy, k, similarity_threshold)
        
        # 3. 加载划分mask
        print("\n📊 Step 3/6: 加载数据划分...")
        train_mask, val_mask, test_mask = self._load_masks()
        
        # 4. 创建PyG Data对象
        print("\n🏗️ Step 4/6: 创建PyG Data对象...")
        data = self._create_pyg_data(X, y, edge_index, edge_weights, 
                                     train_mask, val_mask, test_mask)
        
        # 5. 验证图结构
        print("\n✅ Step 5/6: 验证图结构...")
        self._validate_graph(data)
        
        # 6. 保存
        print("\n💾 Step 6/6: 保存同构图...")
        save_path = self.processed_dir / 'homo_graph.pt'
        torch.save(data, save_path)
        
        # 保存统计信息
        self._save_statistics(strategy, k, similarity_threshold)
        
        print("\n" + "="*70)
        print(f"✅ 同构图构建完成！")
        print(f"📁 已保存: {save_path}")
        print("="*70)
        
        return data
    
    def _load_data(self):
        """加载特征、标签和原始数据"""
        # 特征和标签
        X = np.load(self.processed_dir / 'employee_features.npy')
        y = np.load(self.processed_dir / 'y_turnover_binary.npy')
        
        # 原始数据（用于构建属性边）
        df = pd.read_csv(
            self.raw_dir / 'originaldata.csv',
            encoding='gbk',
            skiprows=1
        )
        
        print(f"   ✓ 员工节点数: {len(X)}")
        print(f"   ✓ 特征维度: {X.shape[1]}")
        print(f"   ✓ 离职员工: {y.sum()} ({y.mean():.2%})")
        print(f"   ✓ 在职员工: {(1-y).sum()} ({(1-y).mean():.2%})")
        
        self.stats['num_nodes'] = len(X)
        self.stats['num_features'] = X.shape[1]
        self.stats['turnover_rate'] = float(y.mean())
        
        return X, y, df
    
    def _load_masks(self):
        """加载训练/验证/测试mask"""
        # 尝试加载.npy文件
        train_mask_path = self.splits_dir / 'train_mask.npy'
        
        if train_mask_path.exists():
            # 如果存在.npy文件，直接加载
            train_mask = np.load(train_mask_path)
            val_mask = np.load(self.splits_dir / 'val_mask.npy')
            test_mask = np.load(self.splits_dir / 'test_mask.npy')
        else:
            # 否则从.pt文件加载
            print("   ℹ️ 从.pt文件加载mask...")
            train_mask = torch.load(self.splits_dir / 'train_mask.pt').numpy()
            val_mask = torch.load(self.splits_dir / 'val_mask.pt').numpy()
            test_mask = torch.load(self.splits_dir / 'test_mask.pt').numpy()
        
        print(f"   ✓ 训练集: {train_mask.sum()} ({train_mask.mean():.1%})")
        print(f"   ✓ 验证集: {val_mask.sum()} ({val_mask.mean():.1%})")
        print(f"   ✓ 测试集: {test_mask.sum()} ({test_mask.mean():.1%})")
        
        return train_mask, val_mask, test_mask
    
    def _build_edges(self, df, X, strategy, k, similarity_threshold):
        """
        构建边索引
        
        Returns:
            edge_index: [2, num_edges]
            edge_weights: [num_edges] 可选的边权重
        """
        if strategy == 'attribute':
            edge_index, edge_weights = self._build_attribute_edges(df)
        elif strategy == 'similarity':
            edge_index, edge_weights = self._build_similarity_edges(X, similarity_threshold)
        elif strategy == 'knn':
            edge_index, edge_weights = self._build_knn_edges(X, k)
        elif strategy == 'hybrid':
            edge_index, edge_weights = self._build_hybrid_edges(df, X, k, similarity_threshold)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        num_edges = edge_index.shape[1]
        avg_degree = num_edges / len(df)
        
        print(f"   ✓ 边数: {num_edges:,}")
        print(f"   ✓ 平均度数: {avg_degree:.2f}")
        print(f"   ✓ 边权重范围: [{edge_weights.min():.3f}, {edge_weights.max():.3f}]")
        
        self.stats['num_edges'] = num_edges
        self.stats['avg_degree'] = float(avg_degree)
        
        return edge_index, edge_weights
    
    def _build_attribute_edges(self, df):
        """
        策略1: 基于共同属性构建边
        
        连接规则:
        - 同岗位类型 → 权重1.0
        - 同公司规模 → 权重0.7
        - 同公司类型 → 权重0.7
        """
        print("   📋 使用属性边策略...")
        
        edges = []
        weights = []
        
        # 修复：使用实际的列名
        post_types = df['Q7'].values
        company_sizes = df['Q8'].values
        company_types = df['Q9'].values
        
        num_nodes = len(df)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_weight = 0.0
                
                # 同岗位
                if post_types[i] == post_types[j]:
                    edge_weight = 1.0
                # 同公司规模
                elif company_sizes[i] == company_sizes[j]:
                    edge_weight = 0.7
                # 同公司类型
                elif company_types[i] == company_types[j]:
                    edge_weight = 0.7
                
                if edge_weight > 0:
                    # 无向图：添加双向边
                    edges.append([i, j])
                    edges.append([j, i])
                    weights.extend([edge_weight, edge_weight])
        
        if not edges:
            print("   ⚠️ 警告: 未找到任何属性边，添加自环")
            edges = [[i, i] for i in range(num_nodes)]
            weights = [1.0] * num_nodes
        
        edge_index = np.array(edges).T
        edge_weights = np.array(weights)
        
        print(f"   ✓ 基于属性的边: {len(edges)}")
        
        return edge_index, edge_weights
    
    def _build_similarity_edges(self, X, threshold):
        """
        策略2: 基于特征相似度构建边
        
        使用余弦相似度，连接相似度 > threshold 的节点对
        """
        print(f"   📏 使用相似度边策略 (阈值={threshold})...")
        
        # 计算余弦相似度矩阵
        similarity = cosine_similarity(X)
        
        # 设置对角线为0（避免自环）
        np.fill_diagonal(similarity, 0)
        
        # 找到相似度 > threshold 的边
        edges = []
        weights = []
        
        num_nodes = X.shape[0]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if similarity[i, j] > threshold:
                    edges.append([i, j])
                    edges.append([j, i])
                    weights.extend([similarity[i, j], similarity[i, j]])
        
        if not edges:
            print(f"   ⚠️ 警告: 阈值{threshold}过高，降低到{threshold-0.1}")
            return self._build_similarity_edges(X, threshold - 0.1)
        
        edge_index = np.array(edges).T
        edge_weights = np.array(weights)
        
        print(f"   ✓ 基于相似度的边: {len(edges)}")
        print(f"   ✓ 平均相似度: {edge_weights.mean():.3f}")
        
        return edge_index, edge_weights
    
    def _build_knn_edges(self, X, k):
        """
        策略3: 基于k近邻构建边
        
        每个节点连接到其k个最近邻居
        """
        print(f"   🎯 使用k-NN边策略 (k={k})...")
        
        # 构建k-NN图
        A = kneighbors_graph(
            X, 
            n_neighbors=k,
            mode='distance',
            include_self=False
        )
        
        # 转换为COO格式
        A_coo = A.tocoo()
        
        # 构建边索引
        edges = np.vstack([A_coo.row, A_coo.col])
        
        # 距离转换为权重（距离越小权重越大）
        distances = A_coo.data
        weights = 1.0 / (1.0 + distances)  # 避免除零
        
        edge_index = edges
        edge_weights = weights
        
        print(f"   ✓ 基于k-NN的边: {edge_index.shape[1]}")
        
        return edge_index, edge_weights
    
    def _build_hybrid_edges(self, df, X, k, threshold):
        """
        策略4: 混合策略（推荐）
        
        结合多种边构建方法:
        1. 属性边（权重高）
        2. k-NN边（确保连通性）
        3. 高相似度边（补充）
        """
        print("   🔄 使用混合边策略...")
        
        all_edges = []
        all_weights = []
        
        # 1. 属性边（权重 * 1.5，优先级最高）
        print("   → 添加属性边...")
        attr_edges, attr_weights = self._build_attribute_edges(df)
        all_edges.append(attr_edges)
        all_weights.append(attr_weights * 1.5)
        
        # 2. k-NN边（确保连通性）
        print(f"   → 添加k-NN边 (k={k})...")
        knn_edges, knn_weights = self._build_knn_edges(X, k)
        all_edges.append(knn_edges)
        all_weights.append(knn_weights)
        
        # 3. 高相似度边（阈值较高，补充语义连接）
        print(f"   → 添加高相似度边 (阈值={threshold})...")
        sim_edges, sim_weights = self._build_similarity_edges(X, threshold)
        all_edges.append(sim_edges)
        all_weights.append(sim_weights)
        
        # 合并所有边
        edge_index = np.hstack(all_edges)
        edge_weights = np.hstack(all_weights)
        
        # 去重（保留最大权重）
        edge_dict = {}
        for idx in range(edge_index.shape[1]):
            src, dst = edge_index[0, idx], edge_index[1, idx]
            weight = edge_weights[idx]
            
            key = (src, dst)
            if key in edge_dict:
                edge_dict[key] = max(edge_dict[key], weight)
            else:
                edge_dict[key] = weight
        
        # 重构
        edges = []
        weights = []
        for (src, dst), weight in edge_dict.items():
            edges.append([src, dst])
            weights.append(weight)
        
        edge_index = np.array(edges).T
        edge_weights = np.array(weights)
        
        print(f"   ✓ 混合策略总边数: {len(edges)}")
        
        return edge_index, edge_weights
    
    def _create_pyg_data(self, X, y, edge_index, edge_weights, 
                        train_mask, val_mask, test_mask):
        """创建PyTorch Geometric Data对象"""
        data = Data(
            x=torch.FloatTensor(X),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_weights).unsqueeze(1),  # [num_edges, 1]
            y=torch.LongTensor(y),
            train_mask=torch.BoolTensor(train_mask),
            val_mask=torch.BoolTensor(val_mask),
            test_mask=torch.BoolTensor(test_mask)
        )
        
        print(f"   ✓ PyG Data对象创建完成")
        print(f"   ✓ 节点特征: {data.x.shape}")
        print(f"   ✓ 边索引: {data.edge_index.shape}")
        print(f"   ✓ 边权重: {data.edge_attr.shape}")
        
        return data
    
    def _validate_graph(self, data):
        """验证图结构的完整性和合理性"""
        print("\n" + "-"*70)
        print("📊 图结构验证")
        print("-"*70)
        
        # 基本信息
        print(f"\n基本信息:")
        print(f"   节点数: {data.num_nodes}")
        print(f"   边数: {data.num_edges}")
        print(f"   特征维度: {data.num_node_features}")
        print(f"   是否有向: {data.is_directed()}")
        
        # 检查自环
        has_self_loops = data.has_self_loops()
        print(f"   是否有自环: {has_self_loops}")
        if has_self_loops:
            print("   ℹ️ 提示: 自环可以帮助保留节点自身信息")
        
        # 检查孤立节点
        has_isolated = data.has_isolated_nodes()
        print(f"   是否有孤立节点: {has_isolated}")
        if has_isolated:
            print("   ⚠️ 警告: 存在孤立节点，可能影响训练")
        
        # 连通性分析
        print(f"\n连通性分析:")
        from torch_geometric.utils import to_networkx
        import networkx as nx
        
        G = to_networkx(data, to_undirected=True)
        is_connected = nx.is_connected(G)
        num_components = nx.number_connected_components(G)
        
        print(f"   是否连通: {is_connected}")
        print(f"   连通分量数: {num_components}")
        
        if not is_connected:
            # 分析各连通分量大小
            components = list(nx.connected_components(G))
            component_sizes = [len(c) for c in components]
            print(f"   最大分量大小: {max(component_sizes)}")
            print(f"   最小分量大小: {min(component_sizes)}")
            print("   ⚠️ 警告: 图不连通，GNN可能无法有效传播信息")
        
        # 度分布
        print(f"\n度分布:")
        degrees = data.edge_index[0].bincount()
        print(f"   最小度: {degrees.min().item()}")
        print(f"   最大度: {degrees.max().item()}")
        print(f"   平均度: {degrees.float().mean().item():.2f}")
        print(f"   中位数度: {degrees.float().median().item():.2f}")
        
        # 边权重分布
        if data.edge_attr is not None:
            print(f"\n边权重分布:")
            weights = data.edge_attr.squeeze()
            print(f"   最小权重: {weights.min().item():.3f}")
            print(f"   最大权重: {weights.max().item():.3f}")
            print(f"   平均权重: {weights.mean().item():.3f}")
            print(f"   中位数权重: {weights.median().item():.3f}")
        
        # 数据划分检查
        print(f"\n数据划分:")
        print(f"   训练集: {data.train_mask.sum().item()} 节点")
        print(f"   验证集: {data.val_mask.sum().item()} 节点")
        print(f"   测试集: {data.test_mask.sum().item()} 节点")
        
        # 标签分布
        print(f"\n标签分布:")
        print(f"   训练集离职率: {data.y[data.train_mask].float().mean().item():.2%}")
        print(f"   验证集离职率: {data.y[data.val_mask].float().mean().item():.2%}")
        print(f"   测试集离职率: {data.y[data.test_mask].float().mean().item():.2%}")
        
        # 保存验证统计
        self.stats.update({
            'is_connected': is_connected,
            'num_components': num_components,
            'has_isolated_nodes': has_isolated,
            'min_degree': int(degrees.min().item()),
            'max_degree': int(degrees.max().item()),
            'avg_degree': float(degrees.float().mean().item())
        })
        
        print("-"*70)
    
    def _save_statistics(self, strategy, k, threshold):
        """保存统计信息"""
        self.stats['strategy'] = strategy
        self.stats['k'] = k
        self.stats['similarity_threshold'] = threshold
        
        stats_path = self.processed_dir / 'homo_graph_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n📊 统计信息已保存: {stats_path}")


def visualize_graph(data, save_path='outputs/homo_graph_visualization.png'):
    """
    可视化同构图（可选）
    
    注意: 对于大图可能很慢
    """
    try:
        import matplotlib.pyplot as plt
        from torch_geometric.utils import to_networkx
        import networkx as nx
        
        print("\n🎨 生成图可视化...")
        
        # 转换为NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # 采样（如果节点太多）
        if G.number_of_nodes() > 100:
            print("   节点数过多，随机采样100个节点")
            nodes = list(G.nodes())[:100]
            G = G.subgraph(nodes)
        
        # 绘图
        plt.figure(figsize=(12, 12))
        
        # 布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 节点颜色（基于标签）
        node_colors = ['red' if data.y[i].item() == 1 else 'lightblue' 
                      for i in G.nodes()]
        
        # 绘制
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=300, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        plt.title("Homogeneous Graph Visualization\n(Red=Turnover, Blue=Stay)")
        plt.axis('off')
        plt.tight_layout()
        
        # 保存
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 可视化已保存: {save_path}")
        
    except ImportError:
        print("   ⚠️ 跳过可视化（需要networkx和matplotlib）")
    except Exception as e:
        print(f"   ⚠️ 可视化失败: {e}")


def main():
    """主函数：构建同构图"""
    print("\n" + "="*70)
    print("🚀 同构图构建脚本")
    print("="*70)
    
    # 创建构建器
    builder = HomogeneousGraphBuilder(data_dir='data')
    
    # 构建同构图（使用混合策略）
    data = builder.build(
        strategy='hybrid',      # 推荐使用混合策略
        k=10,                   # k-NN的k值
        similarity_threshold=0.6  # 相似度阈值
    )
    
    # 可选：可视化
    # visualize_graph(data)
    
    # 测试加载
    print("\n🧪 测试加载...")
    loaded_data = torch.load('data/processed/homo_graph.pt')
    print(f"   ✅ 加载成功！节点数={loaded_data.num_nodes}, 边数={loaded_data.num_edges}")
    
    print("\n" + "="*70)
    print("✅ 全部完成！")
    print("="*70)
    print("\n下一步: python src/models/gcn.py")
    
    return data


if __name__ == '__main__':
    main()