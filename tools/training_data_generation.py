import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# 1. 全局配置与物理参数
# ==========================================
INPUT_FILE = "../data/vpp_dataset_3years.csv"  # 确保文件名一致
OUTPUT_FILE = "../data/vpp_training_data.csv"

# IEEE 33 节点基准参数
BASE_MVA = 10.0
V_BASE = 12.66

# 仿真缩放系数
# IEEE 33 基准负荷 ~3.7 MW。
# 设置 4.5 MW 会导致约 20% 的过载，非常适合训练电压控制。
TARGET_PEAK_LOAD_MW = 4.5
RES_BOOST_FACTOR = 3.0  # 3倍新能源带来强反向潮流，非常适合 VPP 训练

# 电池参数 (部署在 Node 32 - IEEE 33 的末端节点，效果最明显)
BAT_CAPACITY_MWH = 5.0  # 10MWh 对 IEEE33 来说有点太大，5MWh 更合理且具备挑战性
BAT_MAX_POWER_MW = 1.0  # 1-2MW 也是合理的范围
BAT_EFFICIENCY = 0.95
INIT_SOC = 0.5
DT_HOURS = 0.25  # 15分钟

# 电压安全范围 (pu)
V_MAX_LIMIT = 1.05
V_MIN_LIMIT = 0.95


# ==========================================
# 2. 物理先验计算 (保持您的逻辑，略微优化)
# ==========================================
def add_physics_priors(df):
    print("正在计算物理先验特征...")
    # 简单的物理特征工程，保持原样即可
    t_diff = df['temperature'] - 25.0
    efficiency_factor = (1 - 0.005 * t_diff).clip(0.8, 1.2)
    df['phys_pv_est'] = df['irradiance'] * efficiency_factor
    df['phys_pv_est'] /= (df['phys_pv_est'].max() + 1e-6)
    return df


# ==========================================
# 3. 环境搭建 (核心修复：Q值与拓扑)
# ==========================================
def setup_ieee33_network():
    net = pn.case33bw()

    # --- 关键修复 1: 保存 P 和 Q 的基准值 ---
    # 只有同时保存 Q，才能在缩放负荷时保持功率因数不变
    net.base_load_p = net.load['p_mw'].copy()
    net.base_load_q = net.load['q_mvar'].copy()
    net.base_total_load = net.base_load_p.sum()

    # 部署光伏 (分布式：选取几条支路末端)
    pv_buses = [10, 15, 20, 25, 30]
    for bus in pv_buses:
        # 确保之前没有重复添加
        if not any(net.sgen.name == f"PV_{bus}"):
            pp.create_sgen(net, bus=bus, p_mw=0, q_mvar=0, name=f"PV_{bus}", type="PV")

    # 部署风电 (集中式：节点 32)
    if not any(net.sgen.name == "Wind_32"):
        pp.create_sgen(net, bus=32, p_mw=0, q_mvar=0, name="Wind_32", type="WP")

    # 部署储能 (节点 32)
    if not any(net.storage.name == "Bat_32"):
        pp.create_storage(net, bus=32, p_mw=0, max_e_mwh=BAT_CAPACITY_MWH, name="Bat_32")

    return net


# ==========================================
# 4. 专家控制器 (核心修复：物理一致性与因果律)
# ==========================================
def run_expert_simulation(df):
    net = setup_ieee33_network()

    # 获取组件索引
    bat_idx = net.storage[net.storage.name == "Bat_32"].index[0]
    pv_idxs = net.sgen[net.sgen.name.str.contains("PV")].index
    wind_idx = net.sgen[net.sgen.name == "Wind_32"].index[0]

    # 计算全局缩放系数
    raw_peak_load = df['load_mw'].max()
    load_scaler = TARGET_PEAK_LOAD_MW / raw_peak_load
    print(f"负荷缩放系数: {load_scaler:.4f} (Raw: {raw_peak_load:.2f} -> Target: {TARGET_PEAK_LOAD_MW})")

    results = {
        "target_load_mw": [],
        "optimal_p_bat": [],
        "optimal_soc": [],
        "v_uncontrolled": [],  # 新增：无控制时的电压（非常重要的特征）
        "v_final": [],  # 控制后的电压
        "violation_flag": []
    }

    current_soc = INIT_SOC

    print("开始专家策略生成 (Loop)...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # --- A. 环境设置 (Mapping) ---
        p_load_total = row['load_mw'] * load_scaler

        # [关键修复 1] 等比例缩放 P 和 Q，保持功率因数恒定
        scale_factor = p_load_total / net.base_total_load
        net.load['p_mw'] = net.base_load_p * scale_factor
        net.load['q_mvar'] = net.base_load_q * scale_factor

        # 设置新能源出力
        p_pv_total = row['pv_mw'] * load_scaler * RES_BOOST_FACTOR
        p_wind_total = row['wind_mw'] * load_scaler * RES_BOOST_FACTOR

        net.sgen.loc[pv_idxs, 'p_mw'] = p_pv_total / len(pv_idxs)
        net.sgen.at[wind_idx, 'p_mw'] = p_wind_total

        # --- B. 第一次潮流：观测当前状态 (Observe) ---
        # 先把电池关掉，看电网原本的样子
        net.storage.at[bat_idx, 'p_mw'] = 0.0
        try:
            pp.runpp(net)
            v_values = net.res_bus.vm_pu
            v_max_raw = v_values.max()
            v_min_raw = v_values.min()
            # 记录无控制电压
            v_uncontrolled_val = v_max_raw if abs(v_max_raw - 1) > abs(v_min_raw - 1) else v_min_raw
        except:
            v_uncontrolled_val = 1.0
            v_max_raw = 1.0
            v_min_raw = 1.0

        # --- C. 计算物理约束 (关键修复 2: Pre-check Limits) ---
        # 在决定动作前，必须知道电池能干什么
        # 能量 = SoC * Cap
        # 最大放电功率 (受限于能量) = Energy / dt * efficiency
        # 最大充电功率 (受限于空闲容量) = Empty_Capacity / dt / efficiency

        e_avail_discharge = current_soc * BAT_CAPACITY_MWH
        e_avail_charge = (1.0 - current_soc) * BAT_CAPACITY_MWH

        # 计算该时刻理论上物理允许的最大功率边界
        p_phys_max_dis = min(BAT_MAX_POWER_MW, (e_avail_discharge / DT_HOURS) * BAT_EFFICIENCY)
        p_phys_max_chg = min(BAT_MAX_POWER_MW, (e_avail_charge / DT_HOURS) / BAT_EFFICIENCY)

        # 限制范围: [-p_charge, +p_discharge]
        p_lower_bound = -p_phys_max_chg
        p_upper_bound = p_phys_max_dis

        # --- D. 专家决策 ---
        p_cmd = 0.0
        violation = 0

        # 简单的 P-Droop 控制
        if v_max_raw > V_MAX_LIMIT:
            violation = 1
            # 尝试吸收功率 (充电)
            # 灵敏度假设: 0.01 pu / MW (根据 IEEE33 经验值)
            needed_p = -(v_max_raw - V_MAX_LIMIT) / 0.005
            p_cmd = needed_p
        elif v_min_raw < V_MIN_LIMIT:
            violation = -1
            # 尝试发出功率 (放电)
            needed_p = (V_MIN_LIMIT - v_min_raw) / 0.005
            p_cmd = needed_p
        else:
            # 死区内：尝试缓慢回归 50% SoC
            target_soc = 0.5
            p_cmd = (current_soc - target_soc) * 0.5  # 简单的比例回归

        # --- E. 施加硬约束 (Clipping) ---
        # 这里的 Clip 保证了任何生成的 p_bat_cmd 都是物理上一定能执行的
        optimal_p = np.clip(p_cmd, p_lower_bound, p_upper_bound)

        # --- F. 第二次潮流：执行动作 (Act) ---
        net.storage.at[bat_idx, 'p_mw'] = optimal_p
        try:
            pp.runpp(net)
            v_final_val = net.res_bus.vm_pu.max()  # 记录最终最严重的电压点
            if abs(net.res_bus.vm_pu.min() - 1) > abs(v_final_val - 1):
                v_final_val = net.res_bus.vm_pu.min()
        except:
            v_final_val = v_uncontrolled_val

        # --- G. 更新状态 (State Update) ---
        # p > 0 (放电): SoC 减少
        # p < 0 (充电): SoC 增加
        if optimal_p > 0:
            energy_change = -optimal_p * DT_HOURS / BAT_EFFICIENCY
        else:
            energy_change = -optimal_p * DT_HOURS * BAT_EFFICIENCY

        current_soc += energy_change / BAT_CAPACITY_MWH
        current_soc = np.clip(current_soc, 0.0, 1.0)  # 修正数值误差

        # --- H. 记录数据 ---
        results["target_load_mw"].append(p_load_total)
        results["optimal_p_bat"].append(optimal_p)
        results["optimal_soc"].append(current_soc)
        results["v_uncontrolled"].append(v_uncontrolled_val)
        results["v_final"].append(v_final_val)
        results["violation_flag"].append(violation)

    # 合并
    for key, val in results.items():
        df[key] = val

    return df


# ==========================================
# 5. 主程序
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    print("1. 读取数据...")
    df = pd.read_csv(INPUT_FILE)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    print("2. 增强物理特征...")
    df = add_physics_priors(df)

    print("3. IEEE 33 专家仿真...")
    df_final = run_expert_simulation(df)

    print(f"4. 保存训练数据: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("完成。")


if __name__ == "__main__":
    main()