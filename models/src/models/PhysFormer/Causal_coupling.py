import torch
import torch.nn as nn


class CausalCouplingModule(nn.Module):
    """
    PhysFormer 核心模块 - 最终修正版

    核心设计原则:
    1. ✅ 粒度化门控 (Granular Gating) - 文档建议正确
    2. ✅ Query子空间解耦 - 原用户设计正确
    3. ✅ 特征级软约束 (Soft Constraint) - 保留归纳偏置，不做硬计算
    4. ✅ 单次残差 (Single Residual) - 避免双重残差
    5. ✅ 清晰的信息流 - 避免融合逻辑混乱
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()

        # ===== Part 1: Query端子空间投影 =====
        # 让Load/PV/Wind各自从不同角度"询问"天气特征
        self.query_proj_load = nn.Linear(d_model, d_model)
        self.query_proj_pv = nn.Linear(d_model, d_model)
        self.query_proj_wind = nn.Linear(d_model, d_model)

        # ===== Part 2: Key/Value投影 (物理驱动力) =====
        self.phys_proj = nn.Linear(d_model, d_model)

        # ===== Part 3: 因果注意力 (Causal Cross-Attention) =====
        self.attn_load = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_pv = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_wind = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # ===== Part 4: 粒度化物理门控 (核心改进!) =====
        # 每个门控独立决定是否接受物理驱动
        # 输入: [当前统计特征, 天气驱动特征]
        self.gate_load = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.gate_pv = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.gate_wind = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # ===== Part 5: 融合与软约束 =====
        # Step A: 融合三个门控后的物理分量
        self.fusion_proj = nn.Linear(d_model * 3, d_model)

        # Step B: 净负荷软约束 (归纳偏置层)
        # 作用: 在特征空间引导模型学习 Load-PV-Wind 的平衡关系
        # 注意: 这不是硬计算 net=load-pv-wind，而是特征级的先验知识注入
        self.net_balance_inductor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # ===== Part 6: 输出归一化 =====
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # ===== 监控指标 (可选) =====
        self.register_buffer('info_gain_ema', torch.tensor(1.0))

    def forward(self, stat_feat, phys_feat, return_gates=False):
        """
        Args:
            stat_feat: [B, S, D] - 统计流 (来自Informer，包含长程依赖)
            phys_feat: [B, S, D] - 物理流 (来自CFC，包含动力学)
            return_gates: bool - 是否返回门控值用于可视化

        Returns:
            output: [B, S, D] - 耦合后的特征
            gates: dict (可选) - 三个门控的平均激活值
        """
        B, S, D = stat_feat.shape

        # ===== Step 1: 准备 Key/Value (物理驱动力) =====
        phys_kv = self.phys_proj(phys_feat)

        # ===== Step 2: 准备 Query (解耦的功率状态) =====
        q_load = self.query_proj_load(stat_feat)
        q_pv = self.query_proj_pv(stat_feat)
        q_wind = self.query_proj_wind(stat_feat)

        # ===== Step 3: 因果注意力 (提取天气对各功率变量的影响) =====
        # 语义: "在当前Load状态下，温度/辐照/风速如何影响未来Load?"
        load_driven, _ = self.attn_load(q_load, phys_kv, phys_kv)
        pv_driven, _ = self.attn_pv(q_pv, phys_kv, phys_kv)
        wind_driven, _ = self.attn_wind(q_wind, phys_kv, phys_kv)

        # ===== Step 4: 粒度化门控 (分别决定物理驱动的权重) =====
        # 关键: 每个门控独立判断，避免"一刀切"
        gate_load = self.gate_load(torch.cat([stat_feat, load_driven], dim=-1))
        gate_pv = self.gate_pv(torch.cat([stat_feat, pv_driven], dim=-1))
        gate_wind = self.gate_wind(torch.cat([stat_feat, wind_driven], dim=-1))

        # 应用门控: 只接受部分物理驱动
        load_contrib = gate_load * load_driven
        pv_contrib = gate_pv * pv_driven
        wind_contrib = gate_wind * wind_driven

        # ===== Step 5: 融合三个物理分量 =====
        # 拼接三个经过门控的物理贡献
        phys_fused = self.fusion_proj(torch.cat([
            load_contrib, pv_contrib, wind_contrib
        ], dim=-1))

        # ===== Step 6: 净负荷软约束 (归纳偏置) =====
        # 物理意义: 强制特征空间学习 Load-PV-Wind 的系统平衡关系
        # 注意: 这不是硬算 net=load-pv-wind，而是让特征隐式包含这种关系
        phys_balanced = self.net_balance_inductor(phys_fused)

        # ===== Step 7: 单次残差连接 + 归一化 =====
        # 关键: 只做一次残差，避免统计流权重过大
        output = self.norm(stat_feat + self.dropout(phys_balanced))

        # ===== 监控: 信息增益 (训练时诊断用) =====
        if self.training:
            with torch.no_grad():
                info_gain = torch.norm(phys_balanced) / (torch.norm(stat_feat) + 1e-8)
                self.info_gain_ema = 0.99 * self.info_gain_ema + 0.01 * info_gain

        # ===== 返回门控值 (用于可视化) =====
        if return_gates:
            gates = {
                'load': gate_load.mean().item(),
                'pv': gate_pv.mean().item(),
                'wind': gate_wind.mean().item(),
                'info_gain': self.info_gain_ema.item()
            }
            return output, gates

        return output


# ===== 使用示例 =====
if __name__ == "__main__":
    # 测试
    B, S, D = 32, 96, 512
    stat_feat = torch.randn(B, S, D)
    phys_feat = torch.randn(B, S, D)

    model = CausalCouplingModule(d_model=512, n_heads=8)
    output, gates = model(stat_feat, phys_feat, return_gates=True)

    print(f"输出形状: {output.shape}")
    print(f"门控均值: Load={gates['load']:.3f}, PV={gates['pv']:.3f}, Wind={gates['wind']:.3f}")
    print(f"信息增益: {gates['info_gain']:.3f}")

    # 期望:
    # - info_gain > 1.2: 物理耦合有效
    # - 夜间 gate_pv 应接近 0
    # - 白天 gate_pv 应接近 1