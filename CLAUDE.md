# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 提供在本仓库中工作的指导。

## 持久项目记忆

本仓库遵循 `memory.md` 中的持久项目规则。任何环境、训练、验证、测试或实验运行的指导都必须遵循其中定义的环境和 GPU 约束。

## 项目概述

**PhysFormer** — 一个物理引导的 Transformer 模型，用于虚拟电厂（VPP）多目标负荷预测。模型基于7天历史数据（672个时间步，15分钟分辨率）预测未来24小时（96个时间步）的净注入功率，并分解为负载、光伏和风电组件。核心创新在于将物理约束（功率边界、爬坡率、电池SOC限制）结构性地嵌入到架构中，而非使用后处理的钳制方法。

## 项目结构

```
Soft-phys-CFC-Informer/
├── analysis/                    # 分析脚本和报告
├── checkpoints/                 # 模型检查点
├── configs/                     # 配置文件
│   ├── baselines/              # 基准模型配置
│   ├── drivers/                # 驱动配置（基准测试、消融实验）
│   ├── legacy/                 # 遗留配置
│   └── physformer_*.yaml       # PhysFormer主配置
├── data_processed/              # 处理后的数据
├── data_raw/                    # 原始数据
├── docs/                        # 项目文档
├── downloads/                   # 下载文件
├── paper/                       # 论文材料
├── physformer/                  # 核心代码包
│   ├── data/                   # 数据加载和处理
│   ├── exp/                    # 实验训练模块
│   ├── layers/                 # 共享层定义
│   ├── models/                 # 模型定义
│   ├── pipelines/              # 数据处理管道
│   ├── runner/                 # 命令行运行器
│   └── utils/                  # 工具函数
├── runs/                        # 运行结果和日志
├── scripts/                     # 辅助脚本
├── templates/                   # 模板文件
├── tools/                       # 数据处理工具
├── visualization/               # 可视化脚本
├── CLAUDE.md                    # Claude代码指南
├── memory.md                    # 项目环境约束
├── pyproject.toml              # Python项目配置
├── requirements.txt            # 依赖包列表
├── run.py                      # 统一入口点
└── verify_imports.py           # 导入验证
```

## 安装

```bash
# 推荐的可编辑安装
pip install -e .

# 或仅安装依赖
pip install -r requirements.txt
```

**依赖要求**：
- Python 3.8+
- PyTorch >= 2.3.0
- 完整依赖列表见 `requirements.txt`

## 命令行接口

`run.py` 提供了统一的命令行入口点，支持以下子命令：

| 命令 | 描述 | 主要用途 |
|------|------|----------|
| `build-dataset` | 构建多投资组合数据集 | 数据预处理和数据集创建 |
| `train` | 训练单个实验 | 模型训练和微调 |
| `test` | 测试训练好的模型 | 模型评估和指标计算 |
| `benchmark` | 运行基准测试驱动 | 多模型比较和性能评估 |
| `ablation` | 运行消融实验驱动 | 架构组件重要性分析 |
| `export-forecast` | 导出预测结果为CSV | 结果可视化和后续分析 |
| `validate-powerflow` | 电网潮流验证 | 物理合规性验证 |
| `pipeline` | 运行端到端管道 | 完整工作流执行 |

### 数据集构建 (`build-dataset`)
```bash
python run.py build-dataset \
    --nextgen-dir data_raw/nextgen \
    --act-weather-csv data_raw/era5/act_canberra_hourly.csv \
    --rye-generation-csv data_raw/rye/rye_generation_and_load.csv \
    --rye-weather-csv data_raw/era5/rye_template_hourly.csv \
    --output-dir data_processed/multi_portfolio \
    --portfolio-size 5 \
    --wind-penetration-target 0.15
```
- **输入**：原始NextGen数据、ERA5天气数据、Rye发电数据
- **输出**：多投资组合数据集 (`portfolio_dataset_for_training.csv`)
- **关键参数**：投资组合数量、风电渗透率目标、时间区域

### 训练 (`train`)
```bash
# 基本训练
python run.py train --config configs/physformer_default.yaml

# 两阶段训练：第一阶段（净注入预测）
python run.py train --config configs/physformer_default.yaml --run-name physformer_net_injection

# 两阶段训练：第二阶段（操作拟合微调）
python run.py train --config configs/physformer_operational_fit.yaml \
    --init-from-run runs/physformer_net_injection \
    --run-name physformer_operational_fit

# 参数覆盖示例
python run.py train --config configs/physformer_default.yaml \
    --lr 1e-4 --epochs 100 --batch-size 64 \
    --gpu 0 --num-workers 8 --seed 42
```

### 测试 (`test`)
```bash
python run.py test --config configs/physformer_default.yaml --run-name physformer_net_injection
```
- 加载指定运行的检查点
- 计算测试集指标
- 生成预测结果和可视化

### 基准测试 (`benchmark`)
```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection.yaml
```
- 批量运行多个模型配置
- 使用多个随机种子进行稳定性评估
- 生成汇总报告在 `runs/reports/`

### 消融实验 (`ablation`)
```bash
python run.py ablation --config configs/drivers/physformer_ablation.yaml
```
- 系统性地修改模型架构
- 量化组件重要性
- 支持6种消融变体

### 预测导出 (`export-forecast`)
```bash
python run.py export-forecast \
    --config configs/physformer_default.yaml \
    --run-name physformer_net_injection \
    --include-operational-interface
```
- 导出CSV格式的预测结果
- 可选包含操作接口参数 (`operational_scale`, `operational_bias`)

### 网络验证 (`validate-powerflow`)
```bash
python run.py validate-powerflow \
    --config configs/physformer_default.yaml \
    --mapping-csv templates/network_mapping.csv \
    --run-name physformer_net_injection
```
- 使用pandapower/SimBench进行电网验证
- 验证物理约束：功率平衡、电压限制、热限制
- 需要网络映射CSV文件

### 完整管道 (`pipeline`)
```bash
python run.py pipeline \
    --config configs/physformer_default.yaml \
    --mapping-csv templates/network_mapping.csv
```
- 端到端工作流：数据集构建 → 训练 → 测试 → 验证
- 适用于完整的实验复现

### 常用命令行参数
大多数命令支持以下通用参数：
- `--config`：YAML配置文件路径（必需）
- `--run-name`：显式运行名称
- `--gpu`：GPU ID覆盖
- `--seed`：随机种子覆盖
- `--print-config`：打印有效参数并退出
- `--epochs`：训练周期数覆盖
- `--lr`：学习率覆盖
- `--batch-size`：批量大小覆盖
- `--num-workers`：数据加载器工作进程数覆盖
- `--init-from-run`：从现有运行初始化（用于操作拟合）
```

## 核心架构

### PhysFormer 模型组件 (`physformer/models/`)

- **`physformer.py`** — 主模型。双流架构：统计Transformer编码器 + 物理流，通过组件细化机制融合，CFC时序平滑和有界输出头。
- **`physical_layer.py`** — `ExplicitVPPPhysicalLayer`。计算物理基线：
  - **负载分支**：基于舒适温度的非对称冷热响应，包含日历特征和状态追踪
  - **光伏分支**：辐射-温度转换，可学习缩放和温度系数
  - **风力分支**：平滑切入/额定/切出曲线
  - **电池分支**：充放电拆分，效率参数和容量限制
- **`cfc.py`** — 连续函数RNN（ODE层 via torchdiffeq）。建模物理惯性用于残差预测的时序平滑。
- **`flatten_head.py`** — 高效投影头：`[B, S, D] → [B, P, D]` 通过线性时间维度投影。
- **`causal_coupling.py`** — `PhysicsGuidedCausalCoupling` (PGCC)。统计查询与物理键值之间的多头交叉注意力。

### 组件细化机制
1. **共享查询适配器** (`shared_query_adapter`)
2. **组件特定查询适配器** (`component_query_adapters`)
3. **多组件自注意力细化** (`refinement_attn`)

### 共享层 (`physformer/layers/`)
- **`attention.py`** — ProbAttention, FullAttention (支持RoPE)
- **`embedding.py`** — DataEmbedding, TokenEmbedding, TemporalEmbedding
- **`encoder.py`** — Encoder, EncoderLayer, FeedForward, AttentionLayer
- **`positional.py`** — PositionalEncoding, RoPEPositionalEncoding
- **`revin.py`** — RevIN（可逆实例归一化）

### 基准模型 (`physformer/models/`)
- **主流时序预测模型**：Informer, Autoformer, LSTM, GRU, DLinear, PatchTST, iTransformer
- **先进时序模型**：TIDE, TimeXer, TFT (Temporal Fusion Transformer)
- **训练评估**：通过 `physformer/exp/exp_baseline.py` (`Exp_Baselines`) 统一训练评估
- **配置文件**：每个基准模型在 `configs/baselines/` 目录下有对应的配置文件

## 关键特性

### BPAR（有界物理激活残差）
`output = zero_val + Softplus(raw - zero_val)` — 结构性地防止负功率预测，避免梯度消失的钳制。

### 课程学习
软门控从0（纯物理）线性增加到1，逐渐允许网络覆盖物理先验。这是影响最大的组件（消融实验显示8.7%精度影响）。

### 活动掩码
防止在零输出期间（夜间光伏、无风期）学习虚假模式。

### 两阶段训练范式
1. **Stage A (`net_first`)**：优化净注入预测
2. **Stage B (`operational_fit`)**：操作拟合微调，添加 `operational_scale` 和 `operational_bias` 参数

### 物理约束集成
- 网络爬坡限制
- 电池SOC边界
- 充放电互斥性
- 动态物理惩罚调度

### 置信度与归因
- 组件置信度 (`component_confidence`)
- 影响归因 (`component_attribution`)

## 配置系统

### 默认配置 (`configs/physformer_default.yaml`)
```yaml
model:
  name: PhysFormer
  enc_in: 6
  d_model: 512
  n_heads: 8
  e_layers: 3
  d_ff: 2048
  dropout: 0.10
  use_rope: true

data:
  root_path: ./
  data_path: data_processed/multi_portfolio/portfolio_dataset_for_training.csv
  task_mode: net_injection
  target_cols: [p_vpp_mw]
  known_future_covariate_cols: [temperature, irradiance, wind_speed]
  history_state_cols: [p_battery_mw, e_battery_soc_mwh]
  aux_target_cols: [p_load_mw, p_pv_mw, p_wind_mw, p_battery_mw, e_battery_soc_mwh]
  seq_len: 672    # 7天历史（15分钟分辨率）
  pred_len: 96    # 24小时预测

training:
  training_mode: net_first
  use_aux_supervision: false
  batch_size: 64
  train_epochs: 100
  learning_rate: 1.0e-4
  patience: 25
  use_amp: true
```

### 配置覆盖
```bash
# 覆盖硬件设置
python run.py train --config configs/physformer_default.yaml --gpu 0 --num-workers 8

# 覆盖模型参数
python run.py train --config configs/physformer_default.yaml --d_model 256 --n_heads 4
```

## 训练流程

### 数据准备
1. 使用 `run.py build-dataset` 构建多投资组合数据集
2. 数据集包含半合成VPP数据：基于真实天气和生成负载模式
3. 严格的时间分割和投资组合划分

### 训练执行
1. **初始化**：`python run.py train --config configs/physformer_default.yaml`
2. **监控**：训练日志保存在 `runs/<run_name>/logs/`
3. **检查点**：模型保存到 `runs/<run_name>/checkpoint.pth`
4. **配置**：合并后的配置保存到 `runs/<run_name>/config_merged.yaml`

### 两阶段训练
```bash
# 第一阶段：净注入预测
python run.py train --config configs/physformer_default.yaml --run-name physformer_net_injection

# 第二阶段：操作拟合微调
python run.py train --config configs/physformer_operational_fit.yaml \
    --init-from-run runs/physformer_net_injection \
    --run-name physformer_operational_fit
```

## 评估与验证

### 指标 (`physformer/utils/metrics.py`)
- **标准指标**：MAE, RMSE, RSE, CORR
- **物理合规指标**：边界违反率（BVR）、平均违反大小（MVS）
- **通道感知NRMSE**：防止向大规模通道崩溃

### 损失函数 (`physformer/utils/losses.py`)
- **`PhysLoss`**：预测误差 + 边界违反惩罚（L_BVR）+ 爬坡率违反惩罚（L_RVR）
- **`GateResponseRegularization`**：门控与物理先验之间的Pearson相关性

### 网络验证
- 使用 `pandapower` 和 `SimBench` 进行电网验证
- 验证物理约束（功率平衡、电压限制、热限制）

### 分析脚本 (`analysis/`)
- **`export_portfolio_forecasts.py`**：导出投资组合级别的预测结果
- **`validate_portfolio_powerflow.py`**：投资组合级别的电网潮流验证
- **`summarize_benchmark.py`**：汇总基准测试结果，生成统计报告
- **`summarize_ablation.py`**：汇总消融实验结果，分析组件重要性
- **`PhysFormer_Analysis_Report.md`**：PhysFormer架构分析报告（中文）

## 基准测试与消融实验

### 基准测试驱动
```bash
python run.py benchmark --config configs/drivers/benchmark_net_injection.yaml
```
- **比较模型**：PhysFormer, DLinear, TIDE, TimeXer, TFT（支持更多基准模型）
- **稳定性评估**：每个模型使用多个随机种子运行（默认：[2024, 2025, 2026]）
- **结果生成**：在 `runs/reports/` 目录下生成：
  - `benchmark_summary_raw.csv`：原始结果
  - `benchmark_summary_grouped.csv`：分组统计（均值±标准差）
- **配置驱动**：通过 YAML 配置定义基准测试任务列表、超参数和硬件设置

### 消融实验驱动
```bash
python run.py ablation --config configs/drivers/physformer_ablation.yaml
```
- **评估架构组件的重要性**：系统性地移除或修改模型组件
- **支持的消融实验**：
  1. **无物理流** (`no_phys_stream`): 禁用显式物理层
  2. **仅共享查询** (`shared_query_only`): 禁用组件特定查询适配器
  3. **无电池分支** (`no_battery_branch`): 禁用电池物理建模
  4. **无辅助监督** (`no_aux_supervision`): 禁用组件级辅助监督
  5. **无SOC一致性** (`no_soc_consistency`): 禁用电池SOC一致性约束
  6. **无未来天气** (`no_future_weather`): 禁用未来天气信息
- **结果分析**：量化每个组件对模型性能的影响

## 数据管道

### 数据集特征
- **输入特征**：负载、光伏、风电、温度、辐照度、风速
- **目标**：净注入功率 (`p_vpp_mw`)
- **辅助目标**：组件级功率（负载、光伏、风电、电池）
- **未来协变量**：温度、辐照度、风速
- **历史状态**：电池功率、电池SOC

### 数据处理
- **标准化**：StandardScaler 归一化
- **时间编码**：sin/cos 时间编码（小时、星期、月份）
- **投资组合划分**：多个VPP投资组合的联合预测
- **时间泛化测试**：不同时间段的泛化能力评估

## 环境约束

### 本地环境
- 使用名为 `Soft-phys-CFC-Informer` 的 conda 环境
- 本地硬件为 AMD RX6800，不支持 GPU 加速
- GPU 依赖的工作流需在远程云环境中执行

### 远程GPU训练
```bash
# 本地准备命令和配置
# 用户在远程环境中执行：
python run.py train --config configs/physformer_default.yaml --gpu 0
```

## 关键约定

- **安装**：使用 `pip install -e .` 可编辑安装，避免 `sys.path` 技巧
- **配置**：YAML 配置文件在 `configs/` 中提供默认值；命令行参数覆盖它们
- **数据集**：顺序训练/验证/测试分割（无打乱）
- **检查点**：保存到 `runs/<run_name>/`
- **可视化**：图表保存到 `visualization/output/`
- **分析脚本**：位于 `analysis/` 目录中
- **代码注释**：中英文混合
- **论文材料**：位于 `paper/en/` (英文) 和 `paper/zh/` (中文)

## 故障排除

### 常见问题
1. **GPU内存不足**：减小 `batch_size`，启用梯度累积
2. **训练不稳定**：降低学习率，增加 `warmup_epochs`
3. **验证损失不降**：检查数据分割，确保没有数据泄漏
4. **导入错误**：确保已安装包 `pip install -e .`

### 调试命令
```bash
# 验证导入
python verify_imports.py

# 打印配置（不运行训练）
python run.py train --config configs/physformer_default.yaml --print-config

# 检查数据加载
python -c "from physformer.data.data_factory import PhysFormerDataset; import pandas as pd; print('Data loading OK')"
```

## 扩展与修改

### 添加新基准模型
1. 在 `physformer/models/` 中添加模型定义
2. 在 `physformer/exp/exp_baseline.py` 中注册模型
3. 在 `configs/baselines/` 中创建配置文件

### 修改物理层
- 编辑 `physformer/models/physical_layer.py`
- 更新可学习参数和物理方程
- 调整 `ExplicitVPPPhysicalLayer` 中的分支定义

### 添加新损失项
- 编辑 `physformer/utils/losses.py`
- 实现新的损失函数类
- 在 `PhysLoss` 中集成新项

---

*最后更新：2026年4月9日*  
*基于项目分析结果重写*