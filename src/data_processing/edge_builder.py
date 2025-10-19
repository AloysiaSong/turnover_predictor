"""
图边构建模块
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
from pathlib import Path


class EdgeBuilder:
    """异构图边构建器"""
    
    def __init__(self):
        self.edge_index_dict = {}
        
    def build_employee_post_edges(self, df: pd.DataFrame) -> torch.LongTensor:
        """
        构建员工-岗位类别边
        Employee → PostType
        
        Args:
            df: 原始数据框
            
        Returns:
            edge_index: [2, num_edges] 格式的边索引
        """
        print("\n🔗 构建 Employee → PostType 边...")
        
        edges = []
        post_cols = [f'Q5_{i}' for i in range(1, 14)]
        
        for emp_idx in range(len(df)):
            for post_idx, col in enumerate(post_cols):
                if df.iloc[emp_idx][col] == 1:
                    edges.append([emp_idx, post_idx])
        
        edge_index = torch.LongTensor(edges).t()
        
        print(f"   ✅ 边数: {edge_index.shape[1]}")
        print(f"   ✅ 平均每员工连接: {edge_index.shape[1] / len(df):.2f} 个岗位")
        
        return edge_index
    
    def build_employee_company_edges(self, 
                                      df: pd.DataFrame
                                      ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        构建员工-公司属性边
        Employee → CompanySize
        Employee → CompanyType
        
        Args:
            df: 原始数据框
            
        Returns:
            size_edges: [2, 500] - 员工到公司规模的边
            type_edges: [2, 500] - 员工到公司类型的边
        """
        print("\n🔗 构建 Employee → Company 边...")
        
        # 公司规模映射
        size_mapping = {
            '<50': 0, '50?99': 1, '100?499': 2,
            '500?999': 3, '1000?4999': 4, '5000+': 5
        }
        
        # 公司类型映射
        type_mapping = {
            '民营': 0, '国企': 1, '外资': 2,
            '事业单位': 3, '合资': 4, '其他': 5
        }
        
        # 构建规模边
        size_edges = []
        for emp_idx in range(len(df)):
            size_id = size_mapping[df.iloc[emp_idx]['Q4']]
            size_edges.append([emp_idx, size_id])
        
        size_edge_index = torch.LongTensor(size_edges).t()
        print(f"   ✅ Employee → CompanySize: {size_edge_index.shape[1]} 条边")
        
        # 构建类型边
        type_edges = []
        for emp_idx in range(len(df)):
            type_id = type_mapping[df.iloc[emp_idx]['Q3']]
            type_edges.append([emp_idx, type_id])
        
        type_edge_index = torch.LongTensor(type_edges).t()
        print(f"   ✅ Employee → CompanyType: {type_edge_index.shape[1]} 条边")
        
        return size_edge_index, type_edge_index
    
    def build_preference_edges(self, 
                                preference_pairs: pd.DataFrame
                                ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        构建员工-虚拟岗位偏好边
        Employee → HypotheticalPost (prefer / disprefer)
        
        Args:
            preference_pairs: 偏好对数据框
            
        Returns:
            prefer_edges: [2, 3500] - 员工偏好的虚拟岗位
            disprefer_edges: [2, 3500] - 员工不偏好的虚拟岗位
        """
        print("\n🔗 构建 Employee → HypotheticalPost 偏好边...")
        
        prefer_edges = []
        disprefer_edges = []
        
        for _, row in preference_pairs.iterrows():
            emp_idx = row['employee_idx']
            post_A_id = row['post_A_id']
            post_B_id = row['post_B_id']
            choice = row['choice']
            
            if choice == 0:  # 选择岗位A
                prefer_edges.append([emp_idx, post_A_id])
                disprefer_edges.append([emp_idx, post_B_id])
            else:  # 选择岗位B
                prefer_edges.append([emp_idx, post_B_id])
                disprefer_edges.append([emp_idx, post_A_id])
        
        prefer_edge_index = torch.LongTensor(prefer_edges).t()
        disprefer_edge_index = torch.LongTensor(disprefer_edges).t()
        
        print(f"   ✅ Prefer 边: {prefer_edge_index.shape[1]}")
        print(f"   ✅ Disprefer 边: {disprefer_edge_index.shape[1]}")
        
        return prefer_edge_index, disprefer_edge_index
    
    def build_all_edges(self, 
                        df: pd.DataFrame,
                        preference_pairs: pd.DataFrame = None,
                        use_preference: bool = True) -> Dict:
        """
        构建所有边
        
        Args:
            df: 原始数据框
            preference_pairs: 偏好对数据
            use_preference: 是否使用偏好边
            
        Returns:
            edge_index_dict: {edge_type: edge_index}
        """
        print("\n" + "="*60)
        print("图边构建管道")
        print("="*60)
        
        edge_dict = {}
        
        # 1. Employee → PostType
        emp_post_edges = self.build_employee_post_edges(df)
        edge_dict[('employee', 'works_as', 'post_type')] = emp_post_edges
        
        # 2. Employee → Company
        size_edges, type_edges = self.build_employee_company_edges(df)
        edge_dict[('employee', 'at_size', 'company_size')] = size_edges
        edge_dict[('employee', 'at_type', 'company_type')] = type_edges
        
        # 3. Preference edges (可选)
        if use_preference and preference_pairs is not None:
            prefer_edges, disprefer_edges = self.build_preference_edges(preference_pairs)
            edge_dict[('employee', 'prefer', 'hypothetical_post')] = prefer_edges
            edge_dict[('employee', 'disprefer', 'hypothetical_post')] = disprefer_edges
        
        # 统计
        total_edges = sum(e.shape[1] for e in edge_dict.values())
        print(f"\n✅ 边构建完成！")
        print(f"   - 边类型数: {len(edge_dict)}")
        print(f"   - 总边数: {total_edges}")
        
        self.edge_index_dict = edge_dict
        
        return edge_dict
    
    def print_graph_statistics(self):
        """打印图统计信息"""
        print("\n" + "="*60)
        print("图结构统计")
        print("="*60)
        
        for edge_type, edge_index in self.edge_index_dict.items():
            src, relation, dst = edge_type
            print(f"\n【{src} → {relation} → {dst}】")
            print(f"  边数: {edge_index.shape[1]}")
            print(f"  源节点范围: [{edge_index[0].min()}, {edge_index[0].max()}]")
            print(f"  目标节点范围: [{edge_index[1].min()}, {edge_index[1].max()}]")
            
            # 统计连接度
            src_nodes = edge_index[0].unique()
            dst_nodes = edge_index[1].unique()
            print(f"  源节点数: {len(src_nodes)}")
            print(f"  目标节点数: {len(dst_nodes)}")
            print(f"  平均源节点度: {edge_index.shape[1] / len(src_nodes):.2f}")
    
    def save(self, output_dir: str):
        """保存所有边"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n💾 保存边数据...")
        
        for edge_type, edge_index in self.edge_index_dict.items():
            # 转换边类型名称为文件名
            src, relation, dst = edge_type
            filename = f"{src}_{relation}_{dst}.pt"
            
            torch.save(edge_index, output_dir / filename)
            print(f"   ✅ 已保存: {filename}")
        
        # 保存边类型映射
        edge_types = {
            'edge_types': [str(et) for et in self.edge_index_dict.keys()],
            'num_edge_types': len(self.edge_index_dict),
            'total_edges': sum(e.shape[1] for e in self.edge_index_dict.values())
        }
        
        import json
        with open(output_dir / 'edge_info.json', 'w') as f:
            json.dump(edge_types, f, indent=2)
        
        print(f"\n✅ 所有边已保存到: {output_dir}")
    
    def load(self, input_dir: str):
        """加载所有边"""
        input_dir = Path(input_dir)
        
        print(f"\n📂 加载边数据: {input_dir}")
        
        # 加载边类型信息
        import json
        with open(input_dir / 'edge_info.json', 'r') as f:
            edge_info = json.load(f)
        
        # 加载所有边文件
        for pt_file in input_dir.glob('*.pt'):
            if pt_file.name == 'edge_info.json':
                continue
            
            edge_index = torch.load(pt_file)
            
            # 从文件名解析边类型
            name_parts = pt_file.stem.split('_')
            # 假设格式: src_relation_dst.pt
            if len(name_parts) >= 3:
                src = name_parts[0]
                relation = '_'.join(name_parts[1:-1])
                dst = name_parts[-1]
                edge_type = (src, relation, dst)
                
                self.edge_index_dict[edge_type] = edge_index
                print(f"   ✅ 已加载: {pt_file.name}")
        
        print(f"\n✅ 共加载 {len(self.edge_index_dict)} 种边类型")


def main():
    """演示边构建"""
    import sys
    sys.path.append('/Users/yu/code/code2510/gnn')
    from src.data_processing.load_data import DataLoader
    
    # 加载数据
    print("="*60)
    print("开始构建图边数据")
    print("="*60)
    
    loader = DataLoader('/Users/yu/code/code2510/gnn/data/raw/originaldata.csv')
    df = loader.load()
    
    # 加载偏好对
    preference_pairs = pd.read_csv('/Users/yu/code/code2510/gnn/data/processed/preference_pairs.csv')
    
    # 构建边
    builder = EdgeBuilder()
    edge_dict = builder.build_all_edges(
        df, 
        preference_pairs=preference_pairs,
        use_preference=True
    )
    
    # 打印统计
    builder.print_graph_statistics()
    
    # 保存
    output_dir = '/Users/yu/code/code2510/gnn/data/processed/edges'
    builder.save(output_dir)
    
    # 验证加载
    print("\n" + "="*60)
    print("验证边数据加载")
    print("="*60)
    
    builder2 = EdgeBuilder()
    builder2.load(output_dir)
    
    print("\n✅ 边构建完成！")
    
    # 生成可视化图结构描述
    print("\n" + "="*60)
    print("图结构概览")
    print("="*60)
    
    print("""
    异构图结构:
    
    节点类型:
      • Employee (员工): 500个
      • PostType (岗位类别): 13个
      • CompanySize (公司规模): 6个
      • CompanyType (公司类型): 6个
      • HypotheticalPost (虚拟岗位): 14个
    
    边类型:
      • employee → works_as → post_type
      • employee → at_size → company_size
      • employee → at_type → company_type
      • employee → prefer → hypothetical_post
      • employee → disprefer → hypothetical_post
    
    这个异构图可以用于:
      ✓ HomoGNN: 将所有节点视为同一类型
      ✓ HeteroGNN: 利用节点和边的类型信息
      ✓ 多任务学习: 离职预测 + 岗位偏好
    """)


if __name__ == '__main__':
    main()