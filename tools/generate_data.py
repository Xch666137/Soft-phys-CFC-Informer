import simbench as sb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import warnings
# 过滤掉 SimBench 内部产生的 FutureWarning，让控制台更清爽
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =================配置区域=================
OUTPUT_FILE = "../data/vpp_dataset.csv"
# 选择一个2034年高渗透率场景，适合你的研究
SIMBENCH_CODE = "1-MV-rural--2-sw"
YEAR = 2034  # 用于生成时间戳


# ========================================

def generate_weather_from_power(df):
    """
    【核心数据增强逻辑】
    SimBench 只提供功率。为了满足你的 'Physics Priors' 计算需求，
    我们需要通过物理公式逆向生成合理的气象数据，并加入噪声。
    """
    print("正在通过功率反向生成气象特征...")

    n_samples = len(df)
    np.random.seed(42)  # 保证复现性

    # 1. 反推辐照度 (Irradiance)
    # 逻辑: PV Power ~ Irradiance.
    # 我们假设 PV 曲线归一化后对应 0-1000 W/m2，并加入随机云层遮挡噪声
    # 避免除零，加一个小项
    raw_pv = df['pv_mw'] / df['pv_mw'].max()

    # 基础辐照度 = PV形状 * 1000
    irradiance = raw_pv * 1000
    # 添加高斯噪声 (模拟传感器误差或云层瞬变)，只在有光的时候加
    noise = np.random.normal(0, 50, n_samples) * (raw_pv > 0.05)
    df['irradiance'] = (irradiance + noise).clip(0, 1200)

    # 2. 反推风速 (Wind Speed)
    # 逻辑: P ~ v^3. 所以 v ~ P^(1/3).
    # 假设最大风电出力对应 14 m/s (额定风速)
    raw_wind = df['wind_mw'] / df['wind_mw'].max()

    # 防止负数取根号
    raw_wind = raw_wind.clip(0, 1)

    # 基础风速 (加入切入风速 3m/s 的底数)
    wind_speed = 3.0 + (raw_wind ** (1 / 3)) * (12.0 - 3.0)
    # 当出力极低时，风速可能在 0-3 之间随机
    low_wind_mask = raw_wind < 0.01
    wind_speed[low_wind_mask] = np.random.uniform(0, 3.0, size=low_wind_mask.sum())

    # 添加湍流噪声
    wind_noise = np.random.normal(0, 1.5, n_samples)
    df['wind_speed'] = (wind_speed + wind_noise).clip(0, 25)

    # 3. 生成温度 (Temperature)
    # SimBench 不包含温度，我们需要构造一个符合季节和日变化的温度曲线
    # 季节项: Cosine curve (夏天热冬天冷)
    # 日变化项: 与 PV (太阳) 强相关

    # 时间索引
    day_of_year = df['date'].dt.dayofyear
    hour_of_day = df['date'].dt.hour

    # 季节基准: 1月0度，7月25度
    seasonal_temp = 12.5 - 12.5 * np.cos(2 * np.pi * (day_of_year) / 365)

    # 日变化: 下午2点最热，凌晨4点最冷。利用 sin 函数偏移
    daily_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)

    # 随机波动
    temp_noise = np.random.normal(0, 2, n_samples)

    df['temperature'] = seasonal_temp + daily_temp + temp_noise

    # 稍微修正：光照强的时候温度通常更高 (相关性增强)
    df['temperature'] += df['irradiance'] / 1000 * 3.0

    return df


def main():
    if not os.path.exists("../data"):
        os.makedirs("../data")

    print(f"1. 正在从 SimBench 下载/加载电网数据: {SIMBENCH_CODE}...")
    try:
        # 获取 SimBench 数据对象
        net = sb.get_simbench_net(SIMBENCH_CODE)
    except Exception as e:
        print(f"Error: 无法加载 SimBench. 请确保已安装: pip install simbench. 错误信息: {e}")
        return

    print("2. 提取并聚合 Profiles (Load, PV, Wind)...")
    # ================= 核心提取逻辑 =================

    # --- 1. 处理负荷 (Load) ---
    # 目标：data.profiles['load']
    # 动作：只选包含 '_pload' 的列 (有功功率)，自动忽略 time 和 _qload
    if 'load' in net.profiles:
        raw_load = net.profiles['load']
        # filter(like=...) 是 Pandas 强大的筛选器
        valid_load_cols = raw_load.filter(like='_pload')
        total_load = valid_load_cols.sum(axis=1)
    else:
        raise ValueError("严重错误：未找到 Load 数据！")

    # --- 2. 处理新能源 (Renewables) ---
    # SimBench 将 PV 和 Wind 都放在 'renewables' 表中
    if 'renewables' in net.profiles:
        raw_res = net.profiles['renewables']

        # --- 2.1 提取光伏 (PV) ---
        # 筛选列名包含 'PV' 的列 (如 PV3, PV7)
        pv_cols = raw_res.filter(like='PV')
        if not pv_cols.empty:
            total_pv = pv_cols.sum(axis=1)
        else:
            print("提示: renewables 表中未发现明确的 PV 列，设为 0。")
            total_pv = pd.Series(0, index=total_load.index)

        # --- 2.2 提取风电 (Wind/WP) ---
        # 筛选列名包含 'WP' 的列 (Wind Power, 如 WP4, WP7)
        wind_cols = raw_res.filter(like='WP')
        if not wind_cols.empty:
            total_wind = wind_cols.sum(axis=1)
        else:
            print("提示: renewables 表中未发现明确的 WP (风电) 列，设为 0。")
            total_wind = pd.Series(0, index=total_load.index)

    else:
        print("警告: 未找到 'renewables' 表，PV 和风电将设为 0。")
        total_pv = pd.Series(0, index=total_load.index)
        total_wind = pd.Series(0, index=total_load.index)

    # ===============================================

    # 对齐数据长度 (取最小值，防止不同表长度微小差异)
    min_len = min(len(total_load), len(total_pv), len(total_wind))
    total_load = total_load.iloc[:min_len]
    total_pv = total_pv.iloc[:min_len]
    total_wind = total_wind.iloc[:min_len]

    print(f"数据清洗完毕:")
    print(f" -> 负荷峰值: {total_load.max():.2f} MW")
    print(f" -> 光伏峰值: {total_pv.max():.2f} MW")
    print(f" -> 风电峰值: {total_wind.max():.2f} MW")

    # 构建 DataFrame
    df = pd.DataFrame({
        'load_mw': total_load.values,
        'pv_mw': total_pv.values,
        'wind_mw': total_wind.values
    })

    # 创建时间戳
    start_date = pd.Timestamp(f'{YEAR}-01-01')
    df['date'] = pd.date_range(start=start_date, periods=len(df), freq='15min')

    # 简单清洗负值
    df['load_mw'] = df['load_mw'].clip(lower=0)
    df['pv_mw'] = df['pv_mw'].clip(lower=0)
    df['wind_mw'] = df['wind_mw'].clip(lower=0)

    print("3. 执行物理逆向工程与数据增强...")
    # 调用你原本的函数
    df = generate_weather_from_power(df)

    # 重新排列列顺序
    cols = ['date', 'load_mw', 'pv_mw', 'wind_mw', 'temperature', 'irradiance', 'wind_speed']
    df = df[cols]

    print(f"4. 数据生成完毕，保存至 {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)

    # 可视化验证
    plt.figure(figsize=(12, 6))
    plt.plot(df['load_mw'][:200], label='Load (Active)')
    plt.plot(df['pv_mw'][:200], label='PV')
    plt.plot(df['wind_mw'][:200], label='Wind')
    plt.title("Cleaned SimBench Data for VPP Training")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()