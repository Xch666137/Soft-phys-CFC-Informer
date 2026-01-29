# 混合架构文档 (Hybrid Architecture)

## 项目概述

本文档详细描述了基于LEPA (Liquid-Enhanced ProbSparse Attention) 和物理约束ODE系统的混合编码器-解码器架构，该架构专为虚拟电厂调度决策支持系统设计。

## 架构设计

### 核心价值主张

1. **物理一致性**: 集成电力系统物理约束，确保预测结果符合物理规律
2. **时间动态性**: 利用Liquid CFC的时间常数特性，实现时间感知的注意力机制
3. **计算效率**: 保持O(n log n)复杂度的高效稀疏注意力
4. **端到端训练**: 支持联合优化预测精度和物理约束满足度

### 架构组件

```
输入数据 → [混合嵌入层] → [混合编码器] → [混合解码器] → [输出投影] → 预测结果
              ↓             ↓         ↓
          Liquid CFC    LEPA+ODE   LEPA+ODE
```

## 核心模块

### 1. LEPA (Liquid-Enhanced ProbSparse Attention)

#### 数学公式
LEPA的核心计算基于以下公式：

```
A_LEPA = Softmax(QKᵀ/√dₖ ⊙ M ⊙ L(t))V
```

其中：
- `Q, K, V`: 查询、键、值矩阵
- `M`: 稀疏注意力掩码
- `L(t)`: 时间调制函数，基于Liquid CFC时间常数

#### 时间调制函数
```
L(t) = exp(-(t-τ)²/2σ²)
```

- `τ`: 时间常数（来自Liquid CFC）
- `σ`: 时间窗口宽度（可学习参数）

#### 特性
- **时间一致性验证**: 确保注意力权重的时间连续性
- **稀疏性**: 仅保留√n个最重要的注意力连接
- **物理可解释性**: 时间调制反映系统的动态特性

### 2. 物理约束ODE系统

#### 电力系统约束方程

**功率平衡约束**:
```
ΣP_gen + ΣP_storage = ΣP_load + P_loss
```

**频率动态方程**:
```
dΔf/dt = (P_imbalance - D*Δf) / (2H)
```

**电压稳定性方程**:
```
dV/dt = f(Q, Y_network) - α(V - V_ref)
```

#### 约束类型

| 约束类型 | 边界条件 | 单位 |
|----------|----------|------|
| 电压 | [0.95, 1.05] | p.u. |
| 频率 | [49.5, 50.5] | Hz |
| 功率 | [-150, 150] | MW |

#### 约束损失函数
```
L_constraint = λ₁|Power Balance| + λ₂|Voltage Violation| + λ₃|Frequency Violation| + λ₄|Power Violation|
```

### 3. 混合架构

#### 编码器结构
```
HybridEncoder:
- HybridEmbedding (输入投影 + 位置编码 + 时间编码)
- 多层HybridEncoderLayer:
  - LEPA注意力
  - 物理约束ODE处理
  - 前馈网络
  - 层归一化
```

#### 解码器结构
```
HybridDecoder:
- HybridEmbedding (输入投影 + 位置编码 + 时间编码)
- 多层HybridDecoderLayer:
  - 自注意力 (LEPA)
  - 交叉注意力 (LEPA)
  - 物理约束ODE处理
  - 前馈网络
  - 层归一化
```

## 数据流设计

### 输入格式
```python
{
    'src': [batch_size, src_len, input_dim],      # 源序列
    'tgt': [batch_size, tgt_len, input_dim],      # 目标序列
    'targets': [batch_size, tgt_len, output_dim], # 预测目标
    'src_timestamps': [batch_size, src_len],      # 源时间戳
    'tgt_timestamps': [batch_size, tgt_len],      # 目标时间戳
}
```

### 输出格式
```python
{
    'predictions': [batch_size, tgt_len, output_dim],  # 预测结果
    'constraint_satisfaction': float,                  # 约束满足率
    'attention_weights': dict,                         # 注意力权重
    'constraint_violations': dict,                     # 约束违反信息
}
```

## 训练框架

### 损失函数设计

**总损失**:
```
L_total = L_prediction + α·L_constraint + β·L_regularization
```

**各项损失**:
- `L_prediction`: MSE + MAE 预测误差
- `L_constraint`: 物理约束违反惩罚
- `L_regularization`: L2正则化 + 注意力稀疏性 + 时间平滑性

### 优化策略

#### 1. 基于Lyapunov的学习率调度
```python
class LyapunovScheduler:
    def step(self, loss_history):
        stability = |loss_t - loss_{t-1}| / loss_{t-1}
        if stability > threshold:
            lr = max(lr * decay_factor, min_lr)
```

#### 2. 梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 3. 自适应约束权重
```python
λ_constraint = λ_initial * (1 + constraint_violation_rate)
```

## 性能指标

### 准确性指标
- **MSE**: 均方误差 < 0.05
- **R²**: 决定系数 > 0.96
- **MAE**: 平均绝对误差 < 0.1

### 效率指标
- **推理时间**: < 5ms (批量大小=1)
- **内存使用**: < 200MB (批量大小=1)
- **复杂度**: O(n log n)

### 物理约束指标
- **功率平衡误差**: < 3%
- **电压边界满足率**: > 98%
- **频率边界满足率**: > 98%
- **功率边界满足率**: > 98%

## 使用示例

### 快速开始
```python
from src.models.hybrid_architecture import HybridArchitecture
from src.models.physical_constraints import PhysicalConstraintConfig

# 配置约束系统
constraint_config = PhysicalConstraintConfig(
    base_power=100.0,
    base_voltage=220.0,
    system_frequency=50.0
)

# 创建模型
model = HybridArchitecture(
    input_dim=8,      # 输入特征维度
    output_dim=1,     # 输出维度
    d_model=512,      # 模型维度
    n_heads=8,        # 注意力头数
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
    constraint_config=constraint_config
)

# 前向传播
predictions = model(src, tgt)['predictions']
```

### 高级使用
```python
# 带时间戳的预测
result = model(
    src=src,
    tgt=tgt,
    src_timestamps=src_timestamps,
    tgt_timestamps=tgt_timestamps,
    return_attention=True,
    return_constraints=True
)

# 获取约束信息
print(f"约束满足率: {result['constraint_satisfaction']:.4f}")
print(f"注意力权重形状: {result['encoder_attention'].shape}")
print(f"约束违反: {result['encoder_violations'].shape}")
```

## 配置选项

### 模型配置
```python
model_config = {
    'input_dim': 8,
    'output_dim': 1,
    'd_model': 512,
    'n_heads': 8,
    'n_encoder_layers': 6,
    'n_decoder_layers': 6,
    'd_ff': 2048,
    'dropout': 0.1,
    'max_len': 5000
}
```

### 约束配置
```python
constraint_config = {
    'base_power': 100.0,
    'base_voltage': 220.0,
    'system_frequency': 50.0,
    'damping_coefficient': 0.1,
    'inertia_constant': 5.0,
    'ode_solver': 'dopri5',
    'rtol': 1e-3,
    'atol': 1e-4
}
```

### 训练配置
```python
training_config = {
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_epochs': 100,
    'alpha': 1.0,       # 约束权重
    'beta': 0.1,        # 正则化权重
    'gamma': 0.01,      # 稳定性权重
    'max_grad_norm': 1.0,
    'patience': 5
}
```

## 部署建议

### 硬件要求
- **GPU**: NVIDIA RTX 3060 或更高
- **内存**: 8GB RAM 或更高
- **存储**: 1GB 可用空间

### 软件依赖
```
torch >= 1.9.0
torchdiffeq >= 0.2.2
numpy >= 1.19.0
scipy >= 1.7.0
```

### 生产环境配置
```python
# 生产环境优化
model = HybridArchitecture(
    input_dim=8,
    output_dim=1,
    d_model=256,     # 减小模型规模
    n_heads=4,       # 减少注意力头数
    n_encoder_layers=3,
    n_decoder_layers=3,
    d_ff=512,        # 减小前馈网络
    dropout=0.0      # 关闭dropout
)

# 推理优化
model.eval()
with torch.no_grad():
    predictions = model(src, tgt)['predictions']
```

## 扩展性

### 多区域支持
支持多区域电力系统的联合建模：
```python
# 多区域配置
regions = ['华东', '华北', '华南']
models = {}
for region in regions:
    models[region] = HybridArchitecture(
        input_dim=8,
        output_dim=1,
        constraint_config=PhysicalConstraintConfig(
            base_power=regional_capacity[region]
        )
    )
```

### 多时间尺度
支持不同时间尺度的预测：
- **超短期**: 1-5分钟
- **短期**: 15分钟-1小时
- **中期**: 1-24小时
- **长期**: 1-7天

## 故障排除

### 常见问题

**问题**: 训练不稳定
**解决**: 检查学习率和约束权重，使用Lyapunov调度器

**问题**: 约束违反率过高
**解决**: 增加约束权重α，检查物理约束参数

**问题**: 内存不足
**解决**: 减小batch_size或模型规模，使用梯度累积

**问题**: 推理速度慢
**解决**: 使用模型量化或减小序列长度

### 调试工具
```python
# 约束满足率监控
def monitor_constraints(model, data_loader):
    satisfaction_rates = []
    for batch in data_loader:
        result = model(**batch, return_constraints=True)
        satisfaction_rates.append(result['constraint_satisfaction'])
    return np.mean(satisfaction_rates)

# 注意力可视化
import matplotlib.pyplot as plt

def visualize_attention(attention_weights):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[0, 0].cpu().numpy())
    plt.colorbar()
    plt.title('LEPA Attention Weights')
    plt.show()
```

## 版本历史

- **v1.0.0**: 初始版本，包含LEPA和物理约束集成
- **v1.1.0**: 添加多区域支持和性能优化
- **v1.2.0**: 改进约束处理和训练稳定性