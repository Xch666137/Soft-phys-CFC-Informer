# PhysFormer 架构与核心机制分析报告

## 1. 项目概览

**项目名称**: Soft-phys-CFC-Informer (PhysFormer)
**应用场景**: 虚拟电厂（VPP）的多目标净注入功率（Net Injection）预测。
**核心创新**: 
1. 采用“灰盒”物理层，对负载、光伏、风力、电池进行显式物理建模。
2. 两阶段训练范式（`net_first` + `operational_fit`）。
3. 动态物理约束（如爬坡限制、SOC边界约束等）与课程学习机制。

---

## 2. 核心架构与模型设计

### 2.1 PhysFormer 骨干网络 (physformer/models/physformer.py)
模型以 Transformer 为核心（包含编码器 `Encoder` 与时序展平头 `FlattenHead`），并通过多组件细化机制处理复杂特征：
- **统计特征处理**: 通过 `DataEmbedding` 和 Transformer 编码器处理历史特征（包括净注入、历史天气、电池历史状态），并提取粗粒度未来表征 (`coarse_future`)。
- **组件维度细化 (Refinement)**: 模型将 `coarse_future` 与未来天气/时间上下文投影融合，通过共享或特定组件的查询适配器 (`shared_query_adapter` 或 `component_query_adapters`) 和自注意力层 (`refinement_attn`) 为每个组件提取特定的潜在特征 (`load_latent`, `pv_latent`, 等)。

### 2.2 显式物理层 (ExplicitVPPPhysicalLayer)
在 `physformer/models/physical_layer.py` 中实现了组件级别的灰盒物理先验模型：
- **负载分支 (`_load_branch`)**:
  - 基于舒适温度区间的非对称冷热响应。
  - 具备日历特征和轻量线性潜在状态追踪，并融合基础负载参数。
- **光伏分支**:
  - 直接的辐射-温度转换，带有可学习的缩放比例及温度削减系数。
- **风力分支**:
  - 平滑的切入、额定及切出曲线，模拟风电机组理论发电特性。
- **电池分支 (`_battery_branch`)**:
  - 基于物理公式拆分充电与放电，引入充放电效率 (`eta_charge`, `eta_discharge`) 及容量和功率限制。

骨干网络输出的预测值实际上是基于物理层提供的理论值 (`theory`) 的“残差修正 (`delta` 或 `scale`)”，从而保证预测结果的物理合理性。

---

## 3. 物理约束与损失函数

损失函数的实现在 `physformer/utils/losses.py` 中，采用多任务惩罚确保物理一致性：
- **误差项**: 核心网络 MSE/MAE、组件辅助预测误差 (`operational_strong_loss`, `wind_loss`)。
- **动态惩罚项**: 
  - **网络爬坡限制**: 限制输出净功率的剧烈波动 (`net_ramp_penalty`)。
  - **电池动态与 SOC**: 充电与放电的互斥性惩罚 (`anti_overlap_loss`)，以及电池 SOC 转换和边界惩罚 (`soc_transition_loss`, `soc_bounds_loss`)。
- **课程学习 (Curriculum Learning)**: `_curriculum_weights` 调度器通过随 Epochs 增长动态调整辅助任务 (`aux`) 与物理惩罚项 (`physics`) 的权重，缓解前期训练梯度的不稳定性。

---

## 4. 两阶段训练范式

模型在 `PhysLoss` 中支持两种核心训练模式 (`training_mode`)：
1. **`net_first`**: 主要聚焦于优化目标（净注入），让模型先具备基础的时序预测能力。
2. **`operational_fit`**:
   - 冻结大部分骨干网络参数（在模型定义中 `freeze_backbone_for_operational_fit`）。
   - 启用操作缩放和偏移参数 (`operational_scale`, `operational_bias`)。
   - 损失函数强行加入各组件精度损失与物理惩罚，以实现最终“符合操作标准”的微调。

此外，模型还输出各组件的置信度 (`component_confidence`) 及影响归因 (`component_attribution`) 以辅助诊断和可解释性。

---

## 5. 数据处理工作流

统一入口点为 `run.py`，支持灵活的配置和执行：
- **统一实验管线**: 提供从数据集构建、模型训练测试、基准测试，到最后通过 pandapower 验证预测结果的一站式命令行接口。
- **组件解耦配置**: 支持对物理分支进行消融（例如 `no_phys_stream`, `no_battery_branch`, `no_aux_supervision`），极大地方便了消融实验的快速开展和多投资组合场景（`multi_portfolio`）的泛化测试验证。

## 结论
PhysFormer 提供了一套极其稳健的虚拟电厂时间序列预测范式，其设计在数据驱动的灵活性与物理驱动的严谨性之间取得了出色的平衡，特别适合于需要严格约束的电力调度场景。