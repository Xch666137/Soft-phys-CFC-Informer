from zipfile import Path

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class IEEE33Distributor:
    def __init__(self, pv_buses, wind_bus, noise_level=0.05):
        """
        Args:
            pv_buses: list, 光伏接入的节点列表 (如 [10, 15, ...])
            wind_bus: int, 风电接入的节点 (如 32)
            noise_level: float, 空间分配的随机噪声水平 (默认 5%)
        """
        self.pv_buses = pv_buses
        self.wind_bus = wind_bus
        self.noise_level = noise_level

        # --- IEEE 33 标准基准负荷 (kW) ---
        # 来源于 IEEE 标准数据 (对应节点 0-32)
        # Node 0 是平衡节点，通常无负荷，这里按标准填 0
        self.base_loads_kw = np.array([
            0, 100, 90, 120, 60, 60, 200, 200, 60, 60,
            45, 60, 60, 120, 60, 60, 60, 90, 90, 90,
            90, 90, 90, 420, 420, 60, 60, 60, 120, 200,
            150, 210, 60
        ])

        # 计算负荷权重 (归一化)
        self.load_weights = self.base_loads_kw / self.base_loads_kw.sum()

        # 计算 PV 权重 (假设均分，也可以根据容量加权)
        self.pv_weights = np.zeros(33)
        for bus in self.pv_buses:
            self.pv_weights[bus] = 1.0 / len(self.pv_buses)

        # 计算 Wind 权重 (集中式)
        self.wind_weights = np.zeros(33)
        self.wind_weights[self.wind_bus] = 1.0

    def distribute(self, total_pred_mw):
        """
        将总量预测分发到各个节点，并加入空间异质性扰动
        Args:
            total_pred_mw: [Time, 3] -> [Load_Total, PV_Total, Wind_Total]
        Returns:
            nodal_data: [Time, 33, 3]
        """
        T = total_pred_mw.shape[0]
        N = 33

        # 1. 提取总量 [T, 1]
        load_tot = total_pred_mw[:, 0].reshape(-1, 1)
        pv_tot = total_pred_mw[:, 1].reshape(-1, 1)
        wind_tot = total_pred_mw[:, 2].reshape(-1, 1)

        # 2. 基础分配 (广播权重) [T, N]
        load_dist = load_tot * self.load_weights.reshape(1, -1)
        pv_dist = pv_tot * self.pv_weights.reshape(1, -1)
        wind_dist = wind_tot * self.wind_weights.reshape(1, -1)

        # 3. 注入空间扰动 (Spatial Heterogeneity)
        # 仅对 Load 进行扰动，PV/Wind 通常由气象决定，空间相关性强，扰动较小
        # 生成噪声因子: mean=1, std=noise_level
        if self.noise_level > 0:
            noise = np.random.normal(1.0, self.noise_level, size=(T, N))
            load_dist = load_dist * noise

            # 修正：保证总和不变 (Re-normalize)
            # 如果加上噪声后总负荷变了，需要缩放回去，保证 PhysFormer 预测的总量有效
            current_sum = load_dist.sum(axis=1, keepdims=True)
            load_dist = load_dist * (load_tot / (current_sum + 1e-6))

        # 4. 堆叠输出 [T, 33, 3]
        return np.stack([load_dist, pv_dist, wind_dist], axis=-1)


# ============================================================
#  2. 网络构建 (Setup Network)
# ============================================================
def setup_ieee33_vpp(pv_buses, wind_bus, bat_bus):
    net = pn.case33bw()

    # 保存基准值用于计算 Power Factor
    net.base_load_p = net.load['p_mw'].copy()
    net.base_load_q = net.load['q_mvar'].copy()

    # 计算 tan(phi)
    with np.errstate(divide='ignore', invalid='ignore'):
        net.load_tan_phi = np.nan_to_num(net.base_load_q / net.base_load_p, nan=0.0)

    # 部署 PV (Distributed)
    for bus in pv_buses:
        pp.create_sgen(net, bus=bus, p_mw=0, q_mvar=0, name=f"PV_{bus}", type="PV")

    # 部署 Wind (Centralized)
    pp.create_sgen(net, bus=wind_bus, p_mw=0, q_mvar=0, name=f"Wind_{wind_bus}", type="WP")

    # 部署 Battery (VPP Storage)
    pp.create_storage(net, bus=bat_bus, p_mw=0, max_e_mwh=20.0, soc_percent=50.0, name=f"Bat_{bat_bus}")

    return net


# ============================================================
#  3. Simulation (核心修改：电压反馈控制)
# ============================================================
def run_simulation(net, nodal_data, pv_buses, wind_bus, bat_bus, enable_control=True):
    results = {
        'voltage_node_32': [],
        'battery_power': [],
        'soc': []
    }

    timesteps = nodal_data.shape[0]
    bat_idx = net.storage[net.storage.name == f"Bat_{bat_bus}"].index
    bat_capacity = net.storage.max_e_mwh[bat_idx].values[0]
    current_soc = 0.8

    # 限制最大功率 (MW)
    BAT_MAX_P = 1.0

    print(f"Starting Simulation (Control={'ON' if enable_control else 'OFF'})...")

    for t in range(timesteps):
        # --- 1. 设置基础负荷/发电 ---
        all_node_loads_p = nodal_data[t, :, 0]
        valid_load_indices = net.load.bus.values.astype(int)
        mapped_p_loads = all_node_loads_p[valid_load_indices]
        net.load['p_mw'] = mapped_p_loads
        net.load['q_mvar'] = mapped_p_loads * net.load_tan_phi

        for bus in pv_buses:
            sgen_idx = net.sgen[net.sgen.name == f"PV_{bus}"].index
            net.sgen.loc[sgen_idx, 'p_mw'] = nodal_data[t, bus, 1]

        wind_idx = net.sgen[net.sgen.name == f"Wind_{wind_bus}"].index
        net.sgen.loc[wind_idx, 'p_mw'] = nodal_data[t, wind_bus, 2]

        bat_p = 0.0

        # --- 2. 预判电压 (Run without battery) ---
        # 先把电池关掉，看看电网是不是已经不行了
        net.storage.loc[bat_idx, 'p_mw'] = 0.0

        pre_voltage = 1.0
        try:
            pp.runpp(net)
            pre_voltage = net.res_bus.vm_pu[bat_bus]
        except pp.LoadflowNotConverged:
            pre_voltage = 0.8  # 假定严重低压，触发紧急放电

        # --- 3. 电压反馈控制 (Voltage Feedback Control) ---
        if enable_control:
            # 简单的下垂控制 (Droop Control) 思想
            # V < 0.95 -> 放电 (负值)
            # V > 1.05 -> 充电 (正值)

            Kp = 50.0  # 增益系数：电压每偏离 0.01 p.u., 电池出力 0.1 MW

            # 【关键修改 2】提高预警阈值 (0.96 -> 0.98)
            # 含义：不要等跌到谷底才救，跌破 0.98 就开始干预
            target_low = 0.98
            target_high = 1.02  # 同样，防止过压也收紧一点

            if pre_voltage < target_low:
                diff = target_low - pre_voltage
                # 计算需求并限制在最大功率内
                bat_p = -min(diff * Kp, BAT_MAX_P)

            elif pre_voltage > target_high:
                diff = pre_voltage - target_high
                bat_p = min(diff * Kp, BAT_MAX_P)

            # SOC 物理限制检查
            if bat_p < 0 and current_soc <= 0.05:
                bat_p = 0
            elif bat_p > 0 and current_soc >= 0.95:
                bat_p = 0

            energy_change = bat_p * 0.25
            current_soc += energy_change / bat_capacity
            current_soc = np.clip(current_soc, 0.0, 1.0)

            # --- 4. 最终执行 ---
        net.storage.loc[bat_idx, 'p_mw'] = bat_p

        try:
            pp.runpp(net)
            results['voltage_node_32'].append(net.res_bus.vm_pu[bat_bus])
        except pp.LoadflowNotConverged:
            results['voltage_node_32'].append(np.nan)

        results['battery_power'].append(bat_p)
        results['soc'].append(current_soc)

    return results


# ============================================================
#  4. 主执行入口 (Main Execution)
# ============================================================
def main():
    # 1. 加载你的 PhysFormer 预测结果
    base_dir = r"E:\Py_program\Soft-phys-CFC-Informer\exp_results\PhysFormer\checkpoints\PhysFormer_experiment_v2"
    pred_path = os.path.join(base_dir, r"real_prediction.npy")

    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found at {pred_path}")
        preds = np.random.rand(96, 3) * [5, 1, 2]
    else:
        print(f"Loading predictions from: {pred_path}")
        preds = np.load(pred_path)
        # 挑选方差最大的一天 (Stress Day)
        variances = np.var(preds[:, :, 0], axis=1)
        busiest_day_idx = np.argmax(variances)
        preds = preds[busiest_day_idx, :, :]
        print(f"Selected Stress Day Index: {busiest_day_idx}")

    # --- 制造危机：放大系数 1 ---
    STRESS_FACTOR = 1.0
    print(f"Applying Stress Factor: {STRESS_FACTOR}x")
    preds = preds * STRESS_FACTOR

    # 2. 配置拓扑参数
    pv_config = [10, 15, 20, 25, 30]
    wind_config = 32
    bat_config = 32

    # 3. 实例化分发器并执行分发
    print("Distributing predictions to IEEE 33 nodes...")
    distributor = IEEE33Distributor(pv_config, wind_config, noise_level=0.15)
    nodal_data = distributor.distribute(preds)  # [96, 33, 3]

    # 4. 建立网络
    net = setup_ieee33_vpp(pv_config, wind_config, bat_config)

    # 5. 运行仿真 (有控制 vs 无控制)
    res_with_control = run_simulation(net, nodal_data, pv_config, wind_config, bat_config, enable_control=True)
    res_no_control = run_simulation(net, nodal_data, pv_config, wind_config, bat_config, enable_control=False)

    # =======================================================
    # 6. 绘图 (高对比度科研配色版)
    # =======================================================
    plt.figure(figsize=(14, 6))

    # --- 图1：电压对比 (左图) ---
    plt.subplot(1, 2, 1)

    # 1. 画安全区域 (背景)
    plt.fill_between(range(96), 0.95, 1.05, color='gray', alpha=0.15, label='Safe Zone')

    # 2. 画限制线 (黑色虚线)
    plt.axhline(0.95, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    plt.axhline(1.05, color='black', linestyle=':', linewidth=1.5, alpha=0.6)

    # 3. 画 "With Control" (作为底层背景：宽、半透明、实线)
    # 目的：让它像一条“河流”，红线浮在它上面。重合时绿底色依然可见。
    plt.plot(res_with_control['voltage_node_32'],
             color='#2ca02c',  # Tab:Green (科研标准绿)
             linestyle='-',  # 实线
             linewidth=4.0,  # 【关键】线宽设大
             alpha=0.4,  # 【关键】半透明，不遮挡
             label='With PhysFormer Control')

    # 4. 画 "Without Control" (作为顶层前景：细、深色、虚线)
    # 目的：确保在任何时候都能看到红线。
    plt.plot(res_no_control['voltage_node_32'],
             color='#d62728',  # Tab:Red (科研标准红)
             linestyle='--',  # 虚线
             linewidth=2.0,  # 比绿线细
             alpha=1.0,  # 不透明
             label='Without Control')

    plt.title(f'Voltage Stability at Node 32 (Stress Factor={STRESS_FACTOR}x)', fontsize=14, fontweight='bold')
    plt.ylabel('Voltage (p.u.)', fontsize=12)
    plt.xlabel('Time (15min step)', fontsize=12)

    # 优化图例：放在最佳位置，加个框
    plt.legend(loc='lower left', frameon=True, fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- 图2：VPP 调度 (右图) ---
    plt.subplot(1, 2, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 功率曲线 (蓝色实线)
    l1, = ax1.plot(res_with_control['battery_power'],
                   color='#1f77b4',  # Tab:Blue
                   linestyle='-', linewidth=2, label='Battery Power (MW)')

    # SOC 曲线 (橙色实线)
    l2, = ax2.plot(res_with_control['soc'],
                   color='#ff7f0e',  # Tab:Orange
                   linestyle='-', linewidth=2.5, label='SOC')

    ax1.set_ylabel('Power (MW) [-Discharge / +Charge]', color='#1f77b4', fontsize=12, fontweight='bold')
    ax2.set_ylabel('State of Charge (SOC)', color='#ff7f0e', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (15min step)', fontsize=12)

    # 坐标轴颜色匹配
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, 1.1)

    # 合并图例
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, shadow=True)

    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.title('Adaptive VPP Dispatch Strategy', fontsize=14, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(base_dir, 'Fig4_Voltage_Optimization.pdf')
    plt.savefig(save_path, dpi=300)  # 提高分辨率
    print(f"Saved High-Visibility Plot to {save_path}")


if __name__ == '__main__':
    main()