import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """原始数据加载器"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 原始CSV文件路径
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load(self, encoding='gbk', skiprows=1):
        """
        加载CSV数据
        
        Args:
            encoding: 文件编码
            skiprows: 跳过的行数（通常跳过重复表头）
            
        Returns:
            pd.DataFrame: 加载的数据框
        """
        print(f"🔄 正在加载数据: {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path, encoding=encoding, skiprows=skiprows)
            print(f"✅ 成功加载 {len(self.df)} 条样本, {len(self.df.columns)} 个字段")
            return self.df
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            raise
    
    def get_basic_info(self):
        """获取数据集基础信息"""
        if self.df is None:
            raise ValueError("请先调用load()方法加载数据")
        
        info = {
            'n_samples': len(self.df),
            'n_features': len(self.df.columns),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.value_counts().to_dict()
        }
        
        return info
    
    def get_missing_stats(self):
        """获取缺失值统计"""
        if self.df is None:
            raise ValueError("请先调用load()方法加载数据")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        stats = pd.DataFrame({
            'missing_count': missing,
            'missing_pct': missing_pct
        })
        
        return stats[stats['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    def get_column_info(self, column_name: str):
        """获取特定列的详细信息"""
        if self.df is None:
            raise ValueError("请先调用load()方法加载数据")
        
        col = self.df[column_name]
        
        info = {
            'dtype': str(col.dtype),
            'n_unique': col.nunique(),
            'n_missing': col.isnull().sum(),
            'missing_pct': f"{col.isnull().sum() / len(col) * 100:.2f}%"
        }
        
        # 获取值分布
        if col.nunique() < 50:
            info['value_counts'] = col.value_counts().to_dict()
        else:
            info['sample_values'] = col.dropna().head(5).tolist()
        
        # 数值型统计
        if pd.api.types.is_numeric_dtype(col):
            info['statistics'] = {
                'mean': col.mean(),
                'std': col.std(),
                'min': col.min(),
                'max': col.max(),
                'median': col.median()
            }
        
        return info


def main():
    """演示数据加载"""
    # 加载数据
    loader = DataLoader('/Users/yu/code/code2510/gnn/data/raw/originaldata.csv')
    df = loader.load()
    
    # 基础信息
    print("\n" + "="*80)
    print("基础信息")
    print("="*80)
    info = loader.get_basic_info()
    for key, value in info.items():
        if key != 'columns':
            print(f"{key}: {value}")
    
    # 缺失值统计
    print("\n" + "="*80)
    print("缺失值统计")
    print("="*80)
    missing = loader.get_missing_stats()
    if len(missing) > 0:
        print(missing)
    else:
        print("✅ 无缺失值！")
    
    # 查看关键列
    print("\n" + "="*80)
    print("关键列信息")
    print("="*80)
    
    key_columns = ['Q30', 'Q3', 'Q4']
    for col in key_columns:
        print(f"\n【{col}】")
        col_info = loader.get_column_info(col)
        for k, v in col_info.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()