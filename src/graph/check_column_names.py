"""
快速诊断脚本：检查CSV列名
"""
import pandas as pd

# 加载数据
df = pd.read_csv('data/raw/originaldata.csv', encoding='gbk', skiprows=1)

print("\n" + "="*70)
print("CSV列名诊断")
print("="*70)

print(f"\n数据形状: {df.shape}")
print(f"总列数: {len(df.columns)}")

print("\n所有列名:")
print("-"*70)
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

print("\n" + "="*70)
print("查找关键列")
print("="*70)

# 查找岗位相关
print("\n1. 包含'岗位'的列:")
for col in df.columns:
    if '岗位' in col:
        print(f"   ✓ {col}")
        print(f"     前5个值: {df[col].head().tolist()}")

# 查找公司规模
print("\n2. 包含'规模'的列:")
for col in df.columns:
    if '规模' in col:
        print(f"   ✓ {col}")
        print(f"     前5个值: {df[col].head().tolist()}")

# 查找公司类型
print("\n3. 包含'类型'的列:")
for col in df.columns:
    if '类型' in col:
        print(f"   ✓ {col}")
        print(f"     前5个值: {df[col].head().tolist()}")

# 查找Q7-Q9
print("\n4. Q7, Q8, Q9开头的列:")
for col in df.columns:
    if any(col.startswith(q) for q in ['Q7', 'Q8', 'Q9']):
        print(f"   ✓ {col}")

print("\n" + "="*70)
print("建议修复")
print("="*70)

# 生成映射建议
print("\n根据上面的输出，更新homogeneous_graph_builder.py中的列名:")
print("\n修改以下行:")
print("   post_types = df['Q7岗位类型'].values")
print("   company_sizes = df['Q8公司人员规模'].values")
print("   company_types = df['Q9公司类型'].values")
print("\n改为:")
print("   post_types = df['<实际列名1>'].values")
print("   company_sizes = df['<实际列名2>'].values") 
print("   company_types = df['<实际列名3>'].values")

print("\n" + "="*70)
