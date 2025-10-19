"""
准备多任务数据：添加岗位偏好排序
=====================================
从原始CSV中提取7个情景任务的岗位偏好，添加到图数据中
"""

import pandas as pd
import torch
from pathlib import Path
import numpy as np
import re


def prepare_preference_data(
    original_csv='data/raw/originaldata.csv',
    graph_path='data/processed/homo_graph.pt',
    output_path='data/processed/homo_graph_with_preferences.pt'
):
    """
    准备带有岗位偏好的图数据
    
    Args:
        original_csv: 原始数据CSV路径
        graph_path: 现有图数据路径
        output_path: 输出路径
    """
    print("\n" + "="*70)
    print("📋 准备多任务数据：添加岗位偏好")
    print("="*70)
    
    # 1. 加载原始CSV
    print("\n1. 加载原始数据...")
    
    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(original_csv, encoding=encoding)
            print(f"   ✓ 使用编码 {encoding} 成功加载")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"   尝试编码 {encoding} 失败: {e}")
            continue
    
    if df is None:
        raise ValueError("无法用任何编码读取CSV，请检查文件")
    
    print(f"   ✓ 加载 {len(df)} 条记录")
    print(f"   ✓ 列数: {len(df.columns)}")
    
    # 2. 提取岗位偏好列
    print("\n2. 提取岗位偏好数据...")
    
    # 打印所有列名供参考
    print(f"\n   CSV包含 {len(df.columns)} 列")
    print(f"   前20列名称:")
    for i, col in enumerate(df.columns[:20]):
        print(f"      {i+1}. {col}")
    
    if len(df.columns) > 20:
        print(f"   ... (还有 {len(df.columns)-20} 列)")
    
    # 查找偏好相关的列
    preference_cols = []
    
    # 方法1: 直接查找包含情景(S)和岗位(P)的列
    # 例如: S1P1, S1P2, ..., S7P7 或 S1_P1, scenario1_pos1 等
    for scenario_idx in range(1, 8):  # 7个情景
        scenario_patterns = [
            f'S{scenario_idx}',
            f's{scenario_idx}',
            f'scenario{scenario_idx}',
            f'Scenario{scenario_idx}',
            f'情景{scenario_idx}',
        ]
        
        for pattern in scenario_patterns:
            scenario_cols = [col for col in df.columns if pattern in col]
            if len(scenario_cols) > 0:
                preference_cols.extend(scenario_cols)
                break
    
    if len(preference_cols) == 0:
        # 方法2: 查找rank或prefer关键词
        preference_cols = [col for col in df.columns 
                        if 'rank' in col.lower() or 
                            'prefer' in col.lower() or
                            'choice' in col.lower() or
                            'select' in col.lower()]
    
    print(f"\n   找到可能的偏好相关列数: {len(preference_cols)}")
    if len(preference_cols) > 0:
        print(f"   偏好列示例:")
        for col in preference_cols[:10]:
            print(f"      - {col}")
    else:
        print("\n   ⚠️  未找到明确的偏好列")
        print("\n   💡 请检查以下列名，找出岗位偏好相关的列:")
        print("   " + "="*60)
        for i, col in enumerate(df.columns):
            print(f"   {i+1:3d}. {col}")
        print("   " + "="*60)
    
    # 从CSV提取真实偏好数据（尽最大努力）
    print("\n   尝试从CSV提取真实偏好数据...")
    
    n_samples = len(df)
    n_scenarios = 7
    
    # 初始化偏好矩阵 (样本数, 情景数)
    preference_ranks = np.zeros((n_samples, n_scenarios), dtype=int)
    extracted = False
    
    # 优先尝试: 识别“情景选择任务”列（排除岗位信息描述）
    scenario_choice_cols = [
        col for col in df.columns
        if '情景选择任务' in col
        and '岗位信息' not in col
        and any(keyword in col for keyword in ['选择', '请选择', '选择一份', '选择一份你更愿意'])
    ]
    
    def scenario_sort_key(column_name: str) -> int:
        match = re.search(r'任务(\d+)', column_name)
        return int(match.group(1)) if match else 999
    
    scenario_choice_cols = sorted(set(scenario_choice_cols), key=scenario_sort_key)
    
    if len(scenario_choice_cols) >= n_scenarios:
        print(f"   ✓ 找到 {len(scenario_choice_cols)} 个情景选择任务列")
        print(f"   示例列名:")
        for col in scenario_choice_cols[:3]:
            print(f"      - {col}")
        
        for i, col_name in enumerate(scenario_choice_cols[:n_scenarios]):
            col_data = df[col_name].values
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if numeric_data.isna().all():
                    mapping = {
                        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7,
                        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7,
                        '岗位A': 1, '岗位B': 2
                    }
                    numeric_data = pd.Series(col_data).map(mapping).fillna(1).values
                preference_ranks[:, i] = numeric_data.astype(int)
            except Exception:
                print(f"   ⚠️  列 {col_name} 转换失败，使用随机值")
                preference_ranks[:, i] = np.random.randint(1, 8, size=len(df))
        
        extracted = True
        print(f"   ✓ 从情景选择任务提取偏好数据")
    
    # 其次尝试: 假设列名格式为 S1, S2, ..., S7
    if not extracted:
        success = True
        for i in range(n_scenarios):
            col_candidates = [
                f'S{i+1}',
                f's{i+1}',
                f'scenario{i+1}',
                f'Scenario{i+1}',
            ]
            
            found = False
            for col_name in col_candidates:
                if col_name in df.columns:
                    preference_ranks[:, i] = df[col_name].values
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            extracted = True
            print("   ✓ 使用 S{1..7} 模式提取偏好数据")
    
    # 再次尝试: 使用预先收集的匹配列
    if not extracted and len(preference_cols) >= n_scenarios:
        for i, col_name in enumerate(preference_cols[:n_scenarios]):
            preference_ranks[:, i] = df[col_name].values
        extracted = True
        print("   ✓ 使用匹配的偏好列提取数据")
    
    # 校验结果
    if extracted:
        print(f"   ✓ 提取偏好矩阵: {preference_ranks.shape}")
        print(f"   ✓ 数据范围: [{preference_ranks.min()}, {preference_ranks.max()}]")
        
        if preference_ranks.min() < 1 or preference_ranks.max() > 7:
            print(f"   ⚠️  警告: 偏好值不在1-7范围内！")
            print(f"   ⚠️  将使用模拟数据代替")
            extracted = False
    
    # 如果仍未提取成功，则使用模拟数据
    if not extracted:
        print(f"   ⚠️  无法自动提取偏好数据")
        print(f"   🔧 使用模拟数据...")
        
        np.random.seed(42)
        preference_ranks = np.zeros((n_samples, n_scenarios), dtype=int)
        for i in range(n_samples):
            preference_ranks[i] = np.random.permutation(n_scenarios) + 1
        
        print(f"   ✓ 创建模拟偏好矩阵: {preference_ranks.shape}")
    
    # 3. 加载现有图数据
    print("\n3. 加载现有图数据...")
    data = torch.load(graph_path)
    print(f"   ✓ 节点: {data.x.shape[0]}")
    print(f"   ✓ 特征: {data.x.shape[1]}")
    print(f"   ✓ 边: {data.edge_index.shape[1]}")
    
    # 4. 添加偏好数据
    print("\n4. 添加岗位偏好数据到图...")
    
    # 确保样本数匹配（CSV可能比图多1行表头或无效数据）
    if preference_ranks.shape[0] != data.x.shape[0]:
        print(f"   ⚠️  样本数不匹配: CSV={preference_ranks.shape[0]}, Graph={data.x.shape[0]}")
        
        if preference_ranks.shape[0] == data.x.shape[0] + 1:
            print(f"   ✓ CSV比图多1行，移除最后一行")
            preference_ranks = preference_ranks[:-1]
        elif preference_ranks.shape[0] > data.x.shape[0]:
            print(f"   ✓ 截取前{data.x.shape[0]}行")
            preference_ranks = preference_ranks[:data.x.shape[0]]
        else:
            raise ValueError(f"CSV行数({preference_ranks.shape[0]})少于图节点数({data.x.shape[0]})")
    
    # 添加到图数据
    data.preference_ranks = torch.from_numpy(preference_ranks).long()
    
    print(f"   ✓ 添加 preference_ranks: {data.preference_ranks.shape}")
    print(f"   ✓ 数据类型: {data.preference_ranks.dtype}")
    print(f"   ✓ 取值范围: [{data.preference_ranks.min()}, {data.preference_ranks.max()}]")
    
    # 5. 验证数据
    print("\n5. 验证偏好数据...")
    
    # 检查是否所有排序都是1-7
    unique_ranks = torch.unique(data.preference_ranks)
    print(f"   唯一排序值: {unique_ranks.tolist()}")
    
    # 随机抽样展示
    print("\n   随机样本展示:")
    sample_indices = np.random.choice(data.x.shape[0], min(5, data.x.shape[0]), replace=False)
    for idx in sample_indices:
        ranks = data.preference_ranks[idx].numpy()
        print(f"   员工 {idx}: {ranks}")
    
    # 6. 保存
    print(f"\n6. 保存新的图数据...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(data, output_path)
    print(f"   ✓ 保存至: {output_path}")
    
    # 7. 验证保存
    print("\n7. 验证保存的数据...")
    data_loaded = torch.load(output_path)
    assert hasattr(data_loaded, 'preference_ranks'), "preference_ranks丢失"
    print(f"   ✓ 验证成功")
    print(f"   ✓ preference_ranks shape: {data_loaded.preference_ranks.shape}")
    
    print("\n" + "="*70)
    print("✅ 数据准备完成!")
    print("="*70)
    print(f"\n📊 数据统计:")
    print(f"   节点数: {data.x.shape[0]}")
    print(f"   特征维度: {data.x.shape[1]}")
    print(f"   岗位数: {data.preference_ranks.shape[1]}")
    print(f"   边数: {data.edge_index.shape[1]}")
    print(f"\n💾 输出文件: {output_path}")
    print("="*70 + "\n")
    
    return data


def create_mock_preference_data(graph_path, output_path):
    """
    创建模拟的偏好数据（用于快速测试）
    """
    print("\n🔧 创建模拟偏好数据...")
    
    # 加载图
    data = torch.load(graph_path)
    n_samples = data.x.shape[0]
    n_positions = 7
    
    # 生成随机但合理的偏好排序
    np.random.seed(42)
    preference_ranks = np.zeros((n_samples, n_positions), dtype=int)
    
    for i in range(n_samples):
        # 每个员工对7个岗位的随机排序
        preference_ranks[i] = np.random.permutation(n_positions) + 1  # 1-7
    
    # 添加到图
    data.preference_ranks = torch.from_numpy(preference_ranks).long()
    
    # 保存
    torch.save(data, output_path)
    
    print(f"   ✓ 模拟数据已创建: {output_path}")
    print(f"   ✓ preference_ranks: {data.preference_ranks.shape}")
    
    return data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='准备岗位偏好数据')
    parser.add_argument('--mode', type=str, default='real',
                       choices=['real', 'mock'],
                       help='real=从CSV提取, mock=生成模拟数据')
    parser.add_argument('--original-csv', type=str,
                       default='data/raw/originaldata.csv',
                       help='原始CSV路径')
    parser.add_argument('--graph-path', type=str,
                       default='data/processed/homo_graph.pt',
                       help='现有图数据路径')
    parser.add_argument('--output-path', type=str,
                       default='data/processed/homo_graph_with_preferences.pt',
                       help='输出路径')
    
    args = parser.parse_args()
    
    if args.mode == 'real':
        # 从真实CSV提取
        try:
            data = prepare_preference_data(
                args.original_csv,
                args.graph_path,
                args.output_path
            )
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            print("\n💡 提示: 如果CSV列名不明确，请先使用 --mode mock 创建测试数据")
    
    elif args.mode == 'mock':
        # 创建模拟数据
        data = create_mock_preference_data(
            args.graph_path,
            args.output_path
        )
    
    print("\n🎯 下一步:")
    print("   运行多任务训练:")
    print(f"   python scripts/train_gcn_v3.py --data-path {args.output_path}")
