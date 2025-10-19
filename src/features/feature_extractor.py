"""
特征提取模块
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import pickle
from pathlib import Path


class FeatureExtractor:
    """员工特征提取器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_basic_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        提取基础属性特征 (7维)
        
        Args:
            df: 原始数据框
            
        Returns:
            features: (n_samples, 7)
            feature_names: 特征名称列表
        """
        print("📊 提取基础特征...")
        
        features = []
        feature_names = []
        
        # Q6: 总工龄 (数值型)
        q6 = pd.to_numeric(df['Q6'], errors='coerce').fillna(0).values
        features.append(q6)
        feature_names.append('tenure_total')
        
        # Q7: 在岗年限 (数值型)
        q7 = pd.to_numeric(df['Q7'], errors='coerce').fillna(0).values
        features.append(q7)
        feature_names.append('tenure_current')
        
        # Q8: 最近换工作时间 (序数型)
        q8_mapping = {
            '从未': 0, '<1年': 1, '1年': 2, '2年': 3, 
            '3年': 4, '4年': 5, '5年+': 6
        }
        q8 = df['Q8'].map(q8_mapping).fillna(0).values
        features.append(q8)
        feature_names.append('last_job_change')
        
        # Q9: 培训时长 (数值型)
        q9 = pd.to_numeric(df['Q9'], errors='coerce').fillna(0).values
        features.append(q9)
        feature_names.append('training_hours')
        
        # Q10: 通勤时间 (数值型)
        q10 = pd.to_numeric(df['Q10'], errors='coerce').fillna(0).values
        features.append(q10)
        feature_names.append('commute_minutes')
        
        # Q11: 城市满意度 (1-10分)
        q11 = pd.to_numeric(df['Q11'], errors='coerce').fillna(5).values
        features.append(q11)
        feature_names.append('city_satisfaction')
        
        # Q15: 月薪区间 (序数型)
        q15_mapping = {
            '<5k': 0, '5?8k': 1, '8?12k': 2, 
            '12?20k': 3, '20?35k': 4, '35k+': 5
        }
        q15 = df['Q15'].map(q15_mapping).fillna(1).values
        features.append(q15)
        feature_names.append('salary_band')
        
        result = np.column_stack(features)
        print(f"  ✅ 基础特征: {result.shape}")
        
        return result, feature_names
    
    def extract_fit_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        提取人岗匹配度特征 (5维)
        Q12系列: Likert 7分制
        
        Args:
            df: 原始数据框
            
        Returns:
            features: (n_samples, 5)
            feature_names: 特征名称列表
        """
        print("📊 提取人岗匹配度特征...")
        
        likert_mapping = {
            '非常不同意': 1, '不同意': 2, '略不同意': 3, '一般': 4,
            '略同意': 5, '同意': 6, '非常同意': 7
        }
        
        features = []
        feature_names = []
        
        for i in range(1, 6):
            col = f'Q12_{i}'
            values = df[col].map(likert_mapping).fillna(4).values
            features.append(values)
            feature_names.append(f'fit_{i}')
        
        result = np.column_stack(features)
        print(f"  ✅ 人岗匹配特征: {result.shape}")
        
        return result, feature_names
    
    def extract_skill_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        提取技能特征 (30维: 15频率 + 15熟练度)
        Q13系列: 使用频率 (1-5)
        Q14系列: 熟练度 (1-5)
        
        Args:
            df: 原始数据框
            
        Returns:
            features: (n_samples, 30)
            feature_names: 特征名称列表
        """
        print("📊 提取技能特征...")
        
        # 频率映射
        freq_mapping = {
            '几乎不用': 1, '偶尔使用': 2, '一般': 3, 
            '较频繁': 4, '每天高强度': 5
        }
        
        # 熟练度映射
        prof_mapping = {
            '初学': 1, '入门': 2, '熟练': 3, '精通': 4, '专家': 5
        }
        
        features = []
        feature_names = []
        
        # Q13: 使用频率
        for i in range(1, 16):
            col = f'Q13_{i}'
            values = df[col].map(freq_mapping).fillna(3).values
            features.append(values)
            feature_names.append(f'skill_freq_{i}')
        
        # Q14: 熟练度
        for i in range(1, 16):
            col = f'Q14_{i}'
            values = df[col].map(prof_mapping).fillna(3).values
            features.append(values)
            feature_names.append(f'skill_prof_{i}')
        
        result = np.column_stack(features)
        print(f"  ✅ 技能特征: {result.shape}")
        
        return result, feature_names
    
    def extract_economic_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        提取经济损失感知特征 (5维)
        Q16系列: Likert 7分制
        
        Args:
            df: 原始数据框
            
        Returns:
            features: (n_samples, 5)
            feature_names: 特征名称列表
        """
        print("📊 提取经济损失特征...")
        
        likert_mapping = {
            '非常不同意': 1, '不同意': 2, '略不同意': 3, '一般': 4,
            '略同意': 5, '同意': 6, '非常同意': 7
        }
        
        features = []
        feature_names = []
        
        for i in range(1, 6):
            col = f'Q16_{i}'
            values = df[col].map(likert_mapping).fillna(4).values
            features.append(values)
            feature_names.append(f'economic_{i}')
        
        result = np.column_stack(features)
        print(f"  ✅ 经济损失特征: {result.shape}")
        
        return result, feature_names
    
    def extract_all_features(self, df: pd.DataFrame, fit=True) -> Tuple[np.ndarray, List[str]]:
        """
        提取所有特征
        
        Args:
            df: 原始数据框
            fit: 是否拟合标准化器
            
        Returns:
            features: (n_samples, 47) 特征矩阵
            feature_names: 特征名称列表
        """
        print("\n" + "="*60)
        print("特征提取管道")
        print("="*60)
        
        # 提取各组特征
        basic_feats, basic_names = self.extract_basic_features(df)
        fit_feats, fit_names = self.extract_fit_features(df)
        skill_feats, skill_names = self.extract_skill_features(df)
        econ_feats, econ_names = self.extract_economic_features(df)
        
        # 合并特征
        all_features = np.column_stack([
            basic_feats, fit_feats, skill_feats, econ_feats
        ])
        
        all_names = basic_names + fit_names + skill_names + econ_names
        
        print(f"\n📊 特征合并完成: {all_features.shape}")
        print(f"   - 基础特征: {len(basic_names)}维")
        print(f"   - 人岗匹配: {len(fit_names)}维")
        print(f"   - 技能特征: {len(skill_names)}维")
        print(f"   - 经济损失: {len(econ_names)}维")
        print(f"   - 总计: {len(all_names)}维")
        
        # 标准化
        if fit:
            print("\n🔧 拟合标准化器...")
            features_scaled = self.scaler.fit_transform(all_features)
        else:
            print("\n🔧 应用标准化器...")
            features_scaled = self.scaler.transform(all_features)
        
        self.feature_names = all_names
        
        print(f"✅ 特征提取完成！")
        print(f"   - 特征形状: {features_scaled.shape}")
        print(f"   - 特征均值: {features_scaled.mean():.4f}")
        print(f"   - 特征标准差: {features_scaled.std():.4f}")
        
        return features_scaled, all_names
    
    def save(self, path: str):
        """保存特征提取器"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        
        print(f"✅ 特征提取器已保存: {path}")
    
    def load(self, path: str):
        """加载特征提取器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        
        print(f"✅ 特征提取器已加载: {path}")


def main():
    """演示特征提取"""
    import sys
    sys.path.append('/Users/yu/code/code2510/gnn')
    from src.data_processing.load_data import DataLoader
    
    # 加载数据
    loader = DataLoader('/Users/yu/code/code2510/gnn/data/raw/originaldata.csv')
    df = loader.load()
    
    # 提取特征
    extractor = FeatureExtractor()
    features, feature_names = extractor.extract_all_features(df, fit=True)
    
    # 保存特征
    Path('/Users/yu/code/code2510/gnn/data/processed').mkdir(parents=True, exist_ok=True)
    np.save('/Users/yu/code/code2510/gnn/data/processed/employee_features.npy', features)
    with open('/Users/yu/code/code2510/gnn/data/processed/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    # 保存提取器
    Path('/Users/yu/code/code2510/gnn/models').mkdir(parents=True, exist_ok=True)
    extractor.save('/Users/yu/code/code2510/gnn/models/feature_extractor.pkl')
    
    print("\n✅ 特征已保存到 data/processed/")


if __name__ == '__main__':
    main()