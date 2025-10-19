"""
标签提取模块
"""
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


class LabelExtractor:
    """离职标签提取器"""
    
    @staticmethod
    def extract_turnover_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取离职标签
        
        Args:
            df: 原始数据框
            
        Returns:
            y_binary: 二分类标签 (0/1)
            y_prob: 离职概率 (0-1连续值)
        """
        print("\n" + "="*60)
        print("标签提取")
        print("="*60)
        
        # Q30: 3个月离职打算 (二分类)
        print("📌 提取Q30 - 3个月离职打算...")
        y_binary = (df['Q30'] == '会').astype(int).values
        
        pos_count = y_binary.sum()
        neg_count = len(y_binary) - pos_count
        
        print(f"   正样本: {pos_count} ({pos_count/len(y_binary)*100:.1f}%)")
        print(f"   负样本: {neg_count} ({neg_count/len(y_binary)*100:.1f}%)")
        print(f"   不平衡比: {neg_count/pos_count:.1f}:1")
        
        # Q31_1: 6-12个月离职可能性 (回归)
        print("\n📌 提取Q31_1 - 6-12个月离职可能性...")
        y_prob = pd.to_numeric(df['Q31_1'], errors='coerce').fillna(0).values / 100.0
        
        print(f"   范围: {y_prob.min():.2f} - {y_prob.max():.2f}")
        print(f"   均值: {y_prob.mean():.2f}")
        print(f"   中位数: {np.median(y_prob):.2f}")
        
        print("\n✅ 标签提取完成!")
        
        return y_binary, y_prob
    
    @staticmethod
    def extract_preference_pairs(df: pd.DataFrame) -> list:
        """
        提取岗位偏好对
        
        Args:
            df: 原始数据框
            
        Returns:
            preference_pairs: [(emp_idx, task_idx, choice, post_A_id, post_B_id), ...]
        """
        print("\n" + "="*60)
        print("岗位偏好对提取")
        print("="*60)
        
        task_cols = ['Q18', 'Q20', 'Q22', 'Q23', 'Q25', 'Q27', 'Q29']
        preference_pairs = []
        
        for task_idx, q_col in enumerate(task_cols):
            for emp_idx in range(len(df)):
                choice = df.iloc[emp_idx][q_col]
                
                # 编码: 岗位A=0, 岗位B=1
                choice_binary = 0 if choice == '岗位A' else 1
                
                preference_pairs.append({
                    'employee_idx': emp_idx,
                    'task_idx': task_idx,
                    'choice': choice_binary,
                    'post_A_id': task_idx * 2,      # 虚拟岗位ID
                    'post_B_id': task_idx * 2 + 1
                })
        
        print(f"✅ 共提取 {len(preference_pairs)} 个偏好对")
        print(f"   - 7个任务 × {len(df)} 员工 = {7 * len(df)} 对")
        
        return preference_pairs


def main():
    """演示标签提取"""
    import sys
    sys.path.append('/Users/yu/code/code2510/gnn')
    from src.data_processing.load_data import DataLoader
    
    # 加载数据
    loader = DataLoader('/Users/yu/code/code2510/gnn/data/raw/originaldata.csv')
    df = loader.load()
    
    # 提取离职标签
    extractor = LabelExtractor()
    y_binary, y_prob = extractor.extract_turnover_labels(df)
    
    # 保存标签
    output_dir = Path('/Users/yu/code/code2510/gnn/data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'y_turnover_binary.npy', y_binary)
    np.save(output_dir / 'y_turnover_prob.npy', y_prob)
    
    print(f"\n💾 二分类标签已保存: {output_dir / 'y_turnover_binary.npy'}")
    print(f"💾 概率标签已保存: {output_dir / 'y_turnover_prob.npy'}")
    
    # 提取偏好对
    preference_pairs = extractor.extract_preference_pairs(df)
    pref_df = pd.DataFrame(preference_pairs)
    pref_df.to_csv(output_dir / 'preference_pairs.csv', index=False)
    
    print(f"💾 偏好对已保存: {output_dir / 'preference_pairs.csv'}")
    
    # 显示前几行
    print("\n📊 偏好对预览:")
    print(pref_df.head(10))
    
    print("\n✅ 所有标签已保存到 data/processed/")


if __name__ == '__main__':
    main()