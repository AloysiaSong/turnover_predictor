# Week 1-2 快速启动指南 🚀

**目标**: 10-14天完成数据准备和MLP基线模型训练  
**状态**: ✅ 所有代码已准备就绪，可立即开始！

---

## 📦 文件清单

本次交付包含完整的Week 1-2实施代码：

### 1. 文档 (3个)
- `Week1_2_Implementation_Guide.md` - Day 1-6详细指南（环境、特征、边构建）
- `Week1_2_Implementation_Guide_Part2.md` - Day 7-14详细指南（划分、训练、评估）
- `Week1_2_Quick_Start.md` - 本文档（快速启动）

### 2. 核心Python模块 (预计10+个文件)
```
src/
├── data_processing/
│   ├── load_data.py           # 数据加载器
│   ├── label_extractor.py     # 标签提取
│   ├── edge_builder.py        # 边构建器
│   ├── data_splitter.py       # 数据划分
│   └── validate_data.py       # 数据验证
├── features/
│   └── feature_extractor.py   # 特征提取器
├── models/
│   ├── mlp_baseline.py        # MLP模型
│   └── trainer.py             # 训练器
└── evaluation/
    └── evaluator.py           # 评估器
```

### 3. 运行脚本 (3个)
- `train_mlp_baseline.py` - 完整训练脚本（Python）
- `run_baseline.sh` - 一键运行（Linux/Mac）
- `run_baseline.bat` - 一键运行（Windows）

### 4. 配置文件
- `requirements.txt` - Python依赖
- `.gitignore` - Git忽略规则

---

## 🎯 10分钟快速开始

### 方法1: 一键自动化（推荐）

```bash
# 1. 进入项目目录
cd /path/to/your/project

# 2. 复制原始数据
mkdir -p data/raw
cp /path/to/originaldata.csv data/raw/

# 3. 一键运行（自动安装依赖+训练）
# Linux/Mac:
chmod +x run_baseline.sh
./run_baseline.sh

# Windows:
run_baseline.bat
```

### 方法2: 分步手动执行

```bash
# 1. 创建环境
conda create -n hgnn python=3.9
conda activate hgnn

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据
mkdir -p data/raw
cp /path/to/originaldata.csv data/raw/

# 4. 运行训练
python train_mlp_baseline.py
```

**就这么简单！** ⏱️ 预计30分钟-1小时完成训练

---

## 📊 预期结果

### 训练完成后你将得到:

```
✅ 模型文件
   models/mlp/best_model.pt         # 最佳模型checkpoint

✅ 评估报告
   outputs/mlp_baseline/
   ├── evaluation_report.txt        # 文本报告
   ├── results.json                 # JSON结果
   ├── confusion_matrix.png         # 混淆矩阵
   ├── roc_curve.png                # ROC曲线
   ├── pr_curve.png                 # PR曲线
   └── training_history.png         # 训练历史

✅ 处理好的数据
   data/processed/
   ├── employee_features.npy        # 特征矩阵 (500, 47)
   ├── y_turnover_binary.npy        # 标签 (500,)
   └── edges/                       # 图边数据

✅ 数据划分
   data/splits/
   ├── train_idx.npy                # 训练集索引 (340,)
   ├── val_idx.npy                  # 验证集索引 (60,)
   └── test_idx.npy                 # 测试集索引 (100,)
```

### 预期性能指标

```
MLP Baseline (预期范围)
------------------------
ROC AUC:   0.72 - 0.78
F1 Score:  0.35 - 0.45
Precision: 0.40 - 0.55
Recall:    0.30 - 0.45
Accuracy:  0.82 - 0.88
```

*注: 实际结果可能因随机种子和硬件差异略有不同*

---

## 🔧 自定义配置

### 修改模型架构

在 `train_mlp_baseline.py` 中修改：

```python
# 原始（默认）
model = create_mlp_model(
    input_dim=X.shape[1],
    architecture='default',  # [128, 64, 32]
    dropout=0.5
)

# 浅层网络
model = create_mlp_model(
    input_dim=X.shape[1],
    architecture='shallow',  # [64, 32]
    dropout=0.3
)

# 深层网络
model = create_mlp_model(
    input_dim=X.shape[1],
    architecture='deep',     # [256, 128, 64, 32]
    dropout=0.6
)
```

### 修改训练参数

```python
# 在train_mlp_baseline.py中修改
trainer = Trainer(
    model=model,
    device=device,
    learning_rate=0.001,      # 学习率
    weight_decay=1e-4,        # L2正则化
    pos_weight=7.9            # 正样本权重
)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,               # 最大轮数
    early_stopping_patience=15  # 早停轮数
)
```

### 修改数据划分比例

```python
# 在train_mlp_baseline.py中修改
splitter = DataSplitter(
    train_ratio=0.68,  # 68%
    val_ratio=0.12,    # 12%
    test_ratio=0.20,   # 20%
    random_state=42
)
```

---

## 🐛 常见问题排查

### Q1: 找不到originaldata.csv

**问题**:
```
FileNotFoundError: data/raw/originaldata.csv not found
```

**解决**:
```bash
# 检查文件是否存在
ls data/raw/originaldata.csv

# 如果不存在，复制文件
cp /path/to/originaldata.csv data/raw/
```

---

### Q2: 编码错误

**问题**:
```
UnicodeDecodeError: 'utf-8' codec can't decode...
```

**解决**: 代码已自动处理GBK编码，如遇问题检查 `load_data.py`:
```python
df = pd.read_csv(self.data_path, encoding='gbk', skiprows=1)
```

---

### Q3: CUDA/GPU相关错误

**问题**:
```
RuntimeError: CUDA out of memory
```

**解决**:
```python
# 方法1: 减小batch size
train_loader, val_loader, test_loader = create_dataloaders(
    ...,
    batch_size=16  # 原来是32
)

# 方法2: 强制使用CPU
trainer = Trainer(
    model=model,
    device='cpu'  # 而不是'cuda'
)
```

---

### Q4: 依赖包安装失败

**问题**:
```
ERROR: Could not find a version that satisfies...
```

**解决**:
```bash
# 方法1: 更新pip
pip install --upgrade pip

# 方法2: 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方法3: 逐个安装
pip install pandas numpy scikit-learn
pip install torch torchvision
pip install matplotlib seaborn
```

---

### Q5: 训练速度慢

**优化建议**:
```python
# 1. 使用GPU（如果可用）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 增大batch size（GPU内存足够的情况下）
batch_size=64

# 3. 使用更浅的网络
architecture='shallow'

# 4. 减少epoch数
epochs=50
```

---

## 📈 结果解读

### 理解评估报告

运行完成后，查看 `outputs/mlp_baseline/evaluation_report.txt`:

```
=============================================================
模型评估报告
=============================================================

总体指标:
-----------
Accuracy:  0.8500    # 总体准确率
Precision: 0.4545    # 预测为离职的样本中真正离职的比例
Recall:    0.4545    # 所有真实离职样本中被正确预测的比例
F1 Score:  0.4545    # Precision和Recall的调和平均
ROC AUC:   0.7500    # ROC曲线下面积（越接近1越好）
PR AUC:    0.4200    # PR曲线下面积

混淆矩阵:
-----------
              Predicted
              0      1
Actual   0   85      4     # 真负例85，假正例4
         1    6      5     # 假负例6，真正例5
```

**关键指标解释**:

1. **ROC AUC = 0.75**: 
   - 模型区分离职/不离职的能力较好
   - 0.5是随机猜测，1.0是完美分类
   - 0.75表示模型有一定预测能力

2. **F1 = 0.45**:
   - 在不平衡数据集上这是合理的基线
   - GNN模型预期能提升到0.48-0.55

3. **Precision = 0.45**:
   - 预测为"会离职"的样本中，约45%真的离职
   - 对于7.9:1的不平衡数据，这比随机好很多

4. **Recall = 0.45**:
   - 所有真实离职的员工中，模型找到了45%
   - 意味着还有55%被漏掉（假负例）

---

## 🎓 下一步学习路径

### Week 3-4: 图神经网络

完成MLP基线后，开始构建GNN模型：

1. **HomoGNN** (同构图)
   - 所有节点当作同一类型
   - 使用GraphSAGE/GAT
   - 预期AUC提升到0.76-0.80

2. **HeteroGNN** (异构图)
   - 区分员工/岗位/公司节点
   - 使用HeteroConv
   - 预期AUC提升到0.80-0.84

3. **HeteroGNN++ (完整版)**
   - 添加虚拟岗位节点
   - 多任务学习（离职+偏好）
   - 预期AUC提升到0.82-0.86

---

## 📚 参考代码结构

### 最小可运行示例

如果你只想快速测试，这是最小代码：

```python
# minimal_example.py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn as nn

# 1. 加载数据（假设已提取）
X = np.load('data/processed/employee_features.npy')
y = np.load('data/processed/y_turnover_binary.npy')

# 2. 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. 转换为Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# 4. 定义模型
model = nn.Sequential(
    nn.Linear(47, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1)
)

# 5. 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.9]))

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    logits = model(X_train).squeeze()
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test).squeeze()
            test_probs = torch.sigmoid(test_logits).numpy()
            auc = roc_auc_score(y_test.numpy(), test_probs)
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Test AUC: {auc:.4f}')

# 6. 最终评估
model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(model(X_test).squeeze()).numpy()
    final_auc = roc_auc_score(y_test.numpy(), test_probs)
print(f'\nFinal Test AUC: {final_auc:.4f}')
```

只需60行代码即可运行！

---

## ✅ Week 1-2 总结

### 你将学会:

1. ✅ **数据处理**
   - 中文CSV加载（GBK编码）
   - 多类型特征提取
   - 类别变量编码
   - 特征标准化

2. ✅ **图数据准备**
   - 异构图边构建
   - 节点特征矩阵
   - 数据划分与验证

3. ✅ **深度学习基础**
   - PyTorch模型定义
   - 训练循环实现
   - 早停与模型保存
   - 类别不平衡处理

4. ✅ **模型评估**
   - 多指标计算
   - 可视化分析
   - 结果报告生成

### 交付物:

- ✅ 干净的特征数据
- ✅ MLP基线模型
- ✅ 完整评估报告
- ✅ 可复现的代码

### 下周预告:

- 🔜 Week 3-4: GNN实现
- 🔜 多任务学习
- 🔜 超参数优化
- 🔜 消融实验

---

## 🆘 获取帮助

### 如果遇到问题:

1. **检查详细文档**
   - `Week1_2_Implementation_Guide.md` - 完整教程
   - `Week1_2_Implementation_Guide_Part2.md` - 进阶内容

2. **查看代码注释**
   - 所有Python文件都有详细注释
   - 每个函数都有docstring

3. **运行单元测试**
   ```bash
   # 测试数据加载
   python src/data_processing/load_data.py
   
   # 测试特征提取
   python src/features/feature_extractor.py
   
   # 测试模型
   python src/models/mlp_baseline.py
   ```

4. **Debug模式**
   ```bash
   # 使用更多输出信息运行
   python -u train_mlp_baseline.py 2>&1 | tee training.log
   ```

---

## 🎉 准备好了吗？

现在你已经有了：

✅ 完整的实施文档（60+ 页）  
✅ 可运行的Python代码（10+ 模块）  
✅ 一键运行脚本  
✅ 详细的注释和说明  

**立即开始你的Week 1-2之旅吧！** 🚀

```bash
# 让我们开始！
./run_baseline.sh
```

预祝训练顺利！ 💪

---

*最后更新: 2025-10-17*
*作者: AI Research Assistant*
*项目: HGNN离职预测*
