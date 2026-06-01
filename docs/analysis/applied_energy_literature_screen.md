# PhysFormer 的 Applied Energy 文献筛选

日期：2026-06-01

来源：Crossref 公开元数据，`Applied Energy` 的 ISSN 为 `0306-2619`，主要筛选 2018 年以来的期刊论文。当前仅完成题名和元数据层面的初筛；相关性说明由题名推断，下载 PDF 后仍需逐篇核验。

## 筛选目标

构建一个 10-15 篇的核心文献包，用于蒸馏 `Applied Energy` 的期刊画像：

- 该期刊如何表述能源系统层面的研究意义；
- 该期刊通常期待哪些基线、消融、 uncertainty 分析和 operational metrics；
- forecasting 类论文如何把预测精度连接到调度、市场、灵活性或风险价值；
- PhysFormer 是否应该围绕 component-token separation 和 generalization 来叙事，而不是简单宣称“固定物理先验总是有效”。

ARA 约束：

- C11：目前最强的正向结果是 inverted attention 带来的 component-token separation。
- C12：固定 physics-prior 组件虽然改善 Val MSE，却恶化 Test MAE，因此论文不能宣称更强的固定物理先验天然更好。

## 核心下载清单

| ID | 主题桶 | 优先级 | 年份 | 引用数 | DOI | 题名 | 优先下载理由 |
|---|---|---:|---:|---:|---|---|---|
| AE01 | VPP 预测 | 1 | 2024 | 53 | `10.1016/j.apenergy.2024.123890` | Ultra-short-term distributed PV power forecasting for virtual power plant considering data-scarce scenarios | 主题最接近：VPP forecasting、分布式光伏、数据稀缺场景。 |
| AE02 | VPP 综述 | 1 | 2024 | 140 | `10.1016/j.apenergy.2023.122284` | Review of virtual power plant operations: Resource coordination and multidimensional interaction | 可提供 Applied Energy 认可的 VPP 问题框架和运营术语。 |
| AE03 | VPP 不确定性 | 1 | 2019 | 247 | `10.1016/j.apenergy.2019.01.224` | Uncertainties of virtual power plant: Problems and countermeasures | VPP uncertainty 的经典框架，适合支撑引言和预判审稿质疑。 |
| AE04 | VPP 市场/预测误差 | 1 | 2024 | 25 | `10.1016/j.apenergy.2024.124234` | A two-step optimization model for virtual power plant participating in spot market based on energy storage power distribution considering comprehensive forecasting error of renewable energy output | 把 forecasting error 连接到 VPP spot market 参与和储能决策。 |
| AE05 | VPP 爬坡/不确定性 | 1 | 2023 | 31 | `10.1016/j.apenergy.2023.121133` | Distributionally robust comprehensive declaration strategy of virtual power plant participating in the power market considering flexible ramping product and uncertainties | 对 MAE/MSE 之外的 ramping 和 operational-risk 指标很重要。 |
| AE06 | 净负荷预测 | 1 | 2023 | 64 | `10.1016/j.apenergy.2023.120641` | Highly accurate peak and valley prediction short-term net load forecasting approach based on decomposition for power systems with high PV penetration | 与 aggregate net-load forecasting 以及 peak/valley operational value 高度相关。 |
| AE07 | 聚合净负荷 | 1 | 2022 | 32 | `10.1016/j.apenergy.2022.120171` | Aggregated Net-load Forecasting using Markov-Chain Monte-Carlo Regression and C-vine copula | 直接提供 “aggregated net-load forecasting” 锚点和概率/统计基线参考。 |
| AE08 | Transformer 光伏/太阳能 | 1 | 2023 | 142 | `10.1016/j.apenergy.2023.121160` | A Transformer-based multimodal-learning framework using sky images for ultra-short-term solar irradiance forecasting | 可作为 Applied Energy 中 Transformer forecasting 论文的模板：基线、图表、多模态证据。 |
| AE09 | Transformer 风速 | 1 | 2024 | 82 | `10.1016/j.apenergy.2023.122155` | Applicability analysis of transformer to wind speed forecasting by a novel deep learning framework with multiple atmospheric variables | 用于观察 Applied Energy 如何评估 Transformer 的适用性，而不只是看精度。 |
| AE10 | 图网络/Transformer 风速 | 1 | 2023 | 174 | `10.1016/j.apenergy.2022.120565` | Spatio-temporal wind speed forecasting using graph networks and novel Transformer architectures | 图网络与 Transformer 时序预测的强基线/写法参考。 |
| AE11 | 概率 Transformer | 1 | 2025 | 28 | `10.1016/j.apenergy.2025.125294` | Deep probabilistic solar power forecasting with Transformer and Gaussian process approximation | 近期 probabilistic Transformer forecasting 论文，可用于判断不确定性实验预期。 |
| AE12 | Physics-informed 预测 | 1 | 2024 | 36 | `10.1016/j.apenergy.2024.124068` | Physics-informed reinforcement learning for probabilistic wind power forecasting under extreme events | 对 physics-informed claims、robustness 和 extreme-event 叙事很关键。 |
| AE13 | Physics-constrained 预测 | 1 | 2025 | 34 | `10.1016/j.apenergy.2025.125295` | Physics-constrained wind power forecasting aligned with probability distributions for noise-resilient deep learning | 直接关系到 Applied Energy 如何包装 forecasting 中的 physics constraints。 |
| AE14 | Weather-informed 场景 | 1 | 2025 | 26 | `10.1016/j.apenergy.2025.125369` | Weather-informed probabilistic forecasting and scenario generation in power systems | 用于补充 scenario generation、uncertainty 和 weather-informed forecasting 标准。 |
| AE15 | 预测到调度 | 1 | 2024 | 57 | `10.1016/j.apenergy.2024.123548` | Optimal scheduling of renewable energy microgrids: A robust multi-objective approach with machine learning-based probabilistic forecasting | 把 probabilistic forecasting 连接到 scheduling 和 operational value。 |

## 备选文献池

如果核心文献包第一轮阅读后仍存在缺口，再下载这些文献。

| ID | 主题桶 | 年份 | 引用数 | DOI | 题名 | 适用场景 |
|---|---|---:|---:|---|---|---|
| AE16 | 综述 | 2021 | 853 | `10.1016/j.apenergy.2021.117766` | A review of wind speed and wind power forecasting with deep neural networks | 广义 DNN forecasting 相关工作和 baseline taxonomy。 |
| AE17 | 负荷预测 | 2021 | 312 | `10.1016/j.apenergy.2020.116177` | Deep learning for load forecasting with smart meter data: Online Adaptive Recurrent Neural Network | Applied Energy 中较强的 load forecasting benchmark / 写法参考。 |
| AE18 | 分布式光伏 | 2021 | 114 | `10.1016/j.apenergy.2021.117704` | A temporal distributed hybrid deep learning model for day-ahead distributed PV power forecasting | DER/PV forecasting 基线和分布式场景参考。 |
| AE19 | 多能负荷 | 2023 | 81 | `10.1016/j.apenergy.2023.121177` | A multi-task learning method for multi-energy load forecasting based on synthesis correlation analysis and load participation factor | Multi-task load forecasting；与 component coupling 和 C08/C11 有关。 |
| AE20 | 充电站净功率 | 2022 | 50 | `10.1016/j.apenergy.2021.118456` | An Edge Computing-oriented Net Power Forecasting for PV-assisted Charging Station: Model Complexity and Forecasting Accuracy Trade-off | 用于 model complexity 与 forecasting accuracy trade-off 叙事。 |
| AE21 | VPP 灵活性管理 | 2024 | 55 | `10.1016/j.apenergy.2024.123998` | Adaptive multi-agent reinforcement learning for flexible resource management in a virtual power plant with dynamic participating multi-energy buildings | VPP flexibility / operation framing。 |
| AE22 | VPP 优化 | 2025 | 34 | `10.1016/j.apenergy.2025.125333` | Deep reinforcement learning based hierarchical energy management for virtual power plant with aggregated multiple heterogeneous microgrids | 异构 VPP 管理；如果强调 portfolio heterogeneity，可作为补充。 |
| AE23 | EV 负荷预测 | 2024 | 67 | `10.1016/j.apenergy.2024.122801` | Electric vehicles load forecasting for day-ahead market participation using machine and deep learning methods | 面向市场参与的 load forecasting 和 baseline 设计。 |
| AE24 | 城市负荷预测 | 2024 | 65 | `10.1016/j.apenergy.2024.124067` | Future energy insights: Time-series and deep learning models for city load forecasting | 通用 deep learning forecasting 写法和数据集讨论。 |
| AE25 | 预测误差补偿 | 2022 | 18 | `10.1016/j.apenergy.2022.118748` | Forecasting error processing techniques and frequency domain decomposition for forecasting error compensation and renewable energy firming in hybrid systems | Error compensation 与 renewable firming；与 signed error / cancellation 叙事相关。 |
| AE26 | 鲁棒 VPP 调度 | 2020 | 146 | `10.1016/j.apenergy.2020.115707` | Robust stochastic optimal dispatching method of multi-energy virtual power plant considering multiple uncertainties | 较早但价值较高的 VPP uncertainty / dispatch 参考。 |
| AE27 | VPP 运行 | 2020 | 50 | `10.1016/j.apenergy.2020.115222` | Interaction-based virtual power plant operation methodology for distribution system operator's voltage management | VPP coordination 的 operational use case。 |

## 抽取模板

每下载一篇 PDF，就按下表抽取信息。目标是形成期刊画像，不是写普通文献综述。

| 字段 | 说明 |
|---|---|
| `paper_id` | AE01 等编号。 |
| `problem_frame` | 这篇文章解决的能源系统问题是什么？ |
| `why_applied_energy` | 它为什么不只是普通 ML forecasting 论文，而适合 Applied Energy？ |
| `novelty_claim` | 创新点来自 architecture、uncertainty handling、operational coupling、dataset，还是 theory？ |
| `dataset_scope` | 单站点/多站点，公开/私有，预测 horizon，时间分辨率，train/test 划分。 |
| `baselines` | Classical、ML、DL、Transformer、physics-informed、operational optimization 等。 |
| `metrics` | 除 MAE/MSE/RMSE 外，是否包含 ramp、peak、probabilistic score、economic cost、dispatch value。 |
| `ablation_norm` | 消融实验数量、维度和严格程度。 |
| `robustness_norm` | 是否覆盖 seeds、季节、极端天气、uncertainty/OOD、data scarcity。 |
| `figure_table_pattern` | 哪些表格/图支撑核心论点？ |
| `reviewer_objections` | 可能的审稿质疑：数据集泛化性、baseline 强度、指标充分性、operational relevance。 |
| `transfer_to_physformer` | 对 PhysFormer 的 Applied Energy 叙事，哪些写法应借鉴，哪些应避免？ |

## 第一轮蒸馏问题

PDF 准备好后，正式写作前先回答这些问题：

1. 在这个方向上，Applied Energy 是否要求 forecasting-only 论文必须提供 operational value metric？
2. 如果有较强消融和 uncertainty/robustness 证据，单一私有 VPP 数据集是否足够？
3. Applied Energy 的 Transformer forecasting 论文中，哪些 baseline 基本不可缺少？
4. C12 能否包装成能源系统层面的 fixed-prior overfitting 发现，还是会被审稿人视为普通 model-selection artifact？
5. 哪种 title/abstract framing 更安全：VPP aggregate forecasting、component-token separation，还是 physics-prior caution？
