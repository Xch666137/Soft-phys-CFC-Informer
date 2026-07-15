# Applied Energy 15 篇核心论文抽取矩阵

日期：2026-06-01

输入 PDF 路径：`C:\Users\Xch\Desktop\Text\Applied Energy`

抽取状态：15 篇 PDF 均为可读文本型 PDF，均成功映射到 AE01-AE15。少数论文的 `Abstract` 标题未被自动规则定位，但正文、指标、实验和结论文本可读，不影响人工蒸馏。

## 文献映射与可读性

| ID | 页数 | 文本字符数 | 质量 | 题名 |
|---|---:|---:|---|---|
| AE01 | 16 | 69704 | good | Ultra-short-term distributed PV power forecasting for virtual power plant considering data-scarce scenarios |
| AE02 | 20 | 139760 | good | Review of virtual power plant operations: Resource coordination and multidimensional interaction |
| AE03 | 17 | 84945 | good | Uncertainties of virtual power plant: Problems and countermeasures |
| AE04 | 16 | 71858 | good | A two-step optimization model for virtual power plant participating in spot market based on energy storage power distribution considering comprehensive forecasting error of renewable energy output |
| AE05 | 19 | 86967 | good | Distributionally robust comprehensive declaration strategy of virtual power plant participating in the power market considering flexible ramping product and uncertainties |
| AE06 | 13 | 77026 | good | Highly accurate peak and valley prediction short-term net load forecasting approach based on decomposition for power systems with high PV penetration |
| AE07 | 17 | 66536 | good | Aggregated Net-load Forecasting using Markov-Chain Monte-Carlo Regression and C-vine copula |
| AE08 | 19 | 88466 | good | A Transformer-based multimodal-learning framework using sky images for ultra-short-term solar irradiance forecasting |
| AE09 | 20 | 83287 | good | Applicability analysis of transformer to wind speed forecasting by a novel deep learning framework with multiple atmospheric variables |
| AE10 | 13 | 75904 | good | Spatio-temporal wind speed forecasting using graph networks and novel Transformer architectures |
| AE11 | 15 | 85847 | good | Deep probabilistic solar power forecasting with Transformer and Gaussian process approximation |
| AE12 | 12 | 51431 | good | Physics-informed reinforcement learning for probabilistic wind power forecasting under extreme events |
| AE13 | 13 | 64775 | good | Physics-constrained wind power forecasting aligned with probability distributions for noise-resilient deep learning |
| AE14 | 17 | 94098 | good | Weather-informed probabilistic forecasting and scenario generation in power systems |
| AE15 | 24 | 111796 | good | Optimal scheduling of renewable energy microgrids: A robust multi-objective approach with machine learning-based probabilistic forecasting |

## 逐篇抽取矩阵

| ID | 问题框架 | 创新类型 | 实验/指标信号 | Operational link | 对 PhysFormer 的迁移含义 |
|---|---|---|---|---|---|
| AE01 | VPP 内部分布式光伏在 data-scarce 区域的超短期预测。 | Adversarial GNN + graph representation + domain adaptation。 | NRMSE、NMAE；多尺度预测；多组 benchmark/ablation，比较 source/target 信息贡献。 | 预测服务 VPP 内部调度、DER 管理和电力市场参与。 | 如果强调 DVPP/portfolio adaptation，AE01 是最强模板：要把跨区域/跨资源迁移写成能源系统痛点，而不是普通 transfer learning。 |
| AE02 | VPP operation 面临 stochastic resources、heterogeneous information、multi-stakeholder interaction。 | 综述型 taxonomy：energy、communication、market 三个视角。 | 不以预测精度为主，重在 operation decision taxonomy 和技术演进。 | VPP 本质是 DER 协调、市场运行、控制通信和多主体博弈。 | 引言应吸收其 VPP 语言：resource coordination、multidimensional interaction、market operation，而不是只说 forecasting accuracy。 |
| AE03 | VPP 的 renewable power、market price、load demand 三类 uncertainty。 | VPP uncertainty 分类、数学描述、优化目标/约束和示范项目综述。 | 比较 deterministic vs uncertainty-aware optimization；强调 profit、可靠性和工具链。 | uncertainty 直接影响 VPP optimization、market trading、ancillary services。 | PhysFormer 的 net-power forecasting 可以用“三类不确定性”定义问题边界：renewable + load + market/operation，而当前工作主要覆盖前两者。 |
| AE04 | VPP 在 spot market 中受 renewable output forecasting error 和 market price 波动影响。 | 两步优化：按 comprehensive forecasting error 分配储能 reserve/arbitrage power。 | CVaR、profit、risk level、trading deviation penalty。 | 预测误差进入 day-ahead/real-time market 决策，影响储能套利、偏差惩罚和风险偏好。 | 论文若投 Applied Energy，最好增加 imbalance/penalty/reserve 类 proxy metric，否则容易被问“预测提升对 VPP 操作有什么价值”。 |
| AE05 | VPP 参与 energy + flexible ramping product 市场时，需要处理 wind uncertainty。 | DRO declaration-dispatching strategy + FRP market design。 | expected profit +20.44%，wind curtailment cost -59.68%；SO/RO/DRO 对比；96 periods。 | 预测不确定性被连接到 ramping product、reserve capacity、market declaration。 | ramp/peak 指标非常重要。我们的 net_ramp_violation 可以升级成 Applied Energy 友好的 ramping-risk 或 flexibility metric。 |
| AE06 | 高 PV penetration 下的 short-term net load peak/valley forecasting。 | Decomposition + ANN/LSTM 类 hybrid forecasting。 | MAE、MAPE；与 BP、LSTM、EMD-ANN 等比较。 | net load forecasting 直接服务 day-ahead dispatch scheduling；peak-valley ramping stress 是核心能源问题。 | PhysFormer 应把 VPP net power 写成 high-renewable net-load forecasting，而不是单纯 VPP regression。Peak/valley error 可作为补强指标。 |
| AE07 | 高 renewable penetration 下，用 aggregated net-load 简化 load/wind/solar 多不确定变量。 | MCMC regression + C-vine copula 的 aggregated NLF。 | MAE、MAPE；ANN、SVR、Grey 等 reference models；direct vs aggregated NLF。 | accurate net-load forecasts 用于 optimal generation scheduling 和 flexibility requirement estimation。 | 支持我们用 `net = load - pv - wind + batt` 作为核心问题定义；但需解释 aggregate accuracy 与 component errors 的关系。 |
| AE08 | sky images + historical GHI 的 ultra-short-term solar irradiance forecasting。 | Informer + Vision Transformer + cross-modality attention。 | NRMSE 4.28% for 10-min ahead；MAE/MAPE/NRMSE；SOTA comparisons。 | solar uncertainty hinders stable/economic grid operation；forecast supports power system operation and energy marketing。 | Transformer 论文要有强 baseline、图表和多源信息消融。Applied Energy 接受 Transformer，但会要求能源场景与模型结构一一对应。 |
| AE09 | 多 atmospheric variables 下，Transformer 是否适合 wind speed forecasting。 | Hybrid deep learning framework + Transformer applicability analysis。 | 与多个 transformer-based/state-of-the-art/baseline models 比较；发现 vanilla Transformer 可能低于 GRU。 | wind forecasting 减少 power supply cost/risk，提升 grid stability 和 wind utilization。 | 对 C12 很有利：Applied Energy 可以接受“Transformer/复杂模型并不总是最好”的负面结论，但必须有严格比较支撑。 |
| AE10 | 多站点 wind speed 的 spatio-temporal forecasting。 | Graph networks + ST-Transformer/ST-Informer/ST-LogSparse。 | persistence baseline；MAE/MSE；多 horizon 比较。 | 物理模型计算成本高，短期局部预测需要高效数据驱动模型。 | Baseline 里至少应有 persistence/naive，以及主流 DL/Transformer。若只有复杂模型互比，容易显得不完整。 |
| AE11 | solar power 的 probabilistic Transformer forecasting。 | Transformer + Gaussian process approximation。 | deterministic: RMSE/MAE；probabilistic: CRPS；比 MC Dropout 降低 CRPS 22.6%/39.7%。 | probabilistic forecasts 支持储能 reserve capacity 和低成本 grid integration。 | 若我们不能做完整 probabilistic forecasting，也至少要承认 deterministic limitation；最好提供 uncertainty/seed/stability 分析。 |
| AE12 | extreme events 下 wind power probabilistic forecasting 数据稀缺、特征不足。 | Physics-informed reinforcement learning + quantile fitting。 | Pinball loss；reliability、skill score；多风场、多 time scale。 | extreme weather 会显著降低预测精度并威胁 power system safety/stability。 | C12 的写法要谨慎：Applied Energy 中 physics-informed 成功案例很多，我们应说“fixed priors may overfit in VPP aggregate setting”，不是否定 physics-informed。 |
| AE13 | noisy wind speed forecast 下的 noise-resilient wind power forecasting。 | Theory-guided/physics-constrained deep learning，通过 wind power curve probability distribution 约束 LSTM。 | MSE baseline；LSTM only vs JS-loss augmented；autoregressive baselines。 | day-ahead wind forecasting 关系到 wind farm market operation 和 utilization。 | 对比 C12 时要区分“可验证物理约束”和“固定架构先验”。我们的负面结果需要定位在 portfolio-specific fixed priors。 |
| AE14 | power systems 中 weather-informed probabilistic forecasting and scenario generation。 | weather data integration + probabilistic forecasting + Gaussian copula scenario generation。 | ARIMA、DeepAR、NLinear、DLinear、TFT、persistence、SAM；RMSE、CRPS、space/time-sum evaluation。 | probabilistic forecasts 支持 risk quantification、unit commitment、economic dispatch、optimal decisions。 | 这是 Applied Energy 的实验强度上限参考：多 baseline、多层级 aggregate evaluation、scenario/uncertainty。我们至少应补 aggregate/portfolio/peak/ramp 层级指标。 |
| AE15 | renewable microgrid scheduling 中预测精度随 horizon 衰减，影响 dispatch cost。 | ML probabilistic forecasting + robust multi-objective optimization。 | MAE、RMSE、R2、CRPS；operating cost 比 traditional MPC 降低 11.5%。 | forecasting 直接进入 robust optimization，决定 scheduling horizon、dispatch cost 和 uncertainty scenarios。 | 如果要冲 Applied Energy，最强补强是把预测误差映射到简化 dispatch/imbalance cost，而不是只报告 MAE/MSE。 |

## 关键词统计信号

该统计基于自动抽取的摘要、指标句和 operational 句，不是全文精确词频，但足够反映第一轮主题密度。

| 关键词 | 覆盖论文数 | 片段出现次数 | 含义 |
|---|---:|---:|---|
| market | 12 | 117 | Applied Energy 的 VPP/forecasting 叙事强烈偏向市场和运行价值。 |
| uncertainty | 11 | 70 | uncertainty 是几乎不可回避的上层问题框架。 |
| scheduling | 10 | 42 | forecasting 常被要求连接到 scheduling/dispatch。 |
| reserve | 10 | 24 | reserve/flexibility 是 VPP 与新能源预测误差的重要落点。 |
| cost | 9 | 47 | operational/economic cost 是预测改进的强解释路径。 |
| MAE | 9 | 54 | MAE 常见，但通常不单独构成 Applied Energy 贡献。 |
| RMSE | 7 | 27 | RMSE/MAE 是基础精度指标。 |
| CRPS | 3 | 13 | probabilistic forecasting 论文常用 CRPS。 |
| ramping | 4 | 12 | ramping 是 VPP/net-load 方向值得补强的指标。 |
| baseline/benchmark | 9 | 40 | 文章普遍显式安排 benchmark 或 baseline comparison。 |
| Transformer | 4 | 26 | Transformer 可以是方法核心，但不是天然优势，需要 suitability/ablation 证明。 |
| probabilistic | 6 | 42 | probabilistic/uncertainty 方向是 Applied Energy 明显偏好的强化点。 |

## 直接结论

1. Applied Energy 的 forecasting 论文通常不会只停在 MAE/MSE。它们会把预测误差连接到 scheduling、dispatch、market、reserve、ramping、risk 或 cost。
2. Transformer 类文章需要强 baseline。至少需要 naive/persistence、传统统计或 ML、LSTM/GRU、以及近期 Transformer/linear baselines 的组合。
3. 负面结果可以写，但必须写成能源系统发现。AE09 已经提供了“Transformer 不一定最适合”的期刊内先例；这支持我们把 C12 写成 fixed-prior overfitting caution。
4. 单一私有 VPP 数据集存在风险。若无法补公开数据集，必须用多 portfolio/多 horizon/多 seed/数据稀缺或 target adaptation 来补外部有效性。
5. PhysFormer 当前最适合的 Applied Energy 主线不是“物理先验越强越好”，而是“VPP aggregate forecasting 中，component-token separation 比固定先验更能稳定泛化；固定先验可能造成 validation-test overfitting”。

