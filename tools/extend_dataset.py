import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "../data/vpp_dataset.csv"
OUTPUT_FILE = "../data/vpp_dataset_3years.csv"


def augment_year(df_origin, year_offset, noise_level=0.05):
    """
    生成一个"平行宇宙"的年份数据。
    保持物理相关性，但加入随机扰动和趋势。
    """
    df_new = df_origin.copy()
    n = len(df_new)

    # 1. 时间平移
    # 假设原始 data 是 2034，我们往后顺延
    df_new['date'] = pd.to_datetime(df_new['date']) + pd.DateOffset(years=year_offset)

    # 2. 负荷增强 (Load Augmentation)
    # 假设每年负荷自然增长 2% (EV 渗透率提高)
    growth_factor = 1.0 + (0.02 * year_offset)
    # 每日随机波动: 模拟用户行为的不确定性 (±5%)
    random_fluctuation = np.random.normal(1.0, noise_level, n)
    df_new['load_mw'] = df_new['load_mw'] * growth_factor * random_fluctuation

    # 3. 光伏与辐照度增强 (PV & Irradiance)
    # 它们必须同步变化 (物理强相关)
    # 模拟"大小年"：比如某年整体雨水多，光照少 5%
    yearly_sun_factor = np.random.uniform(0.95, 1.05)
    # 短时云层噪声
    cloud_noise = np.random.normal(1.0, noise_level, n)

    df_new['pv_mw'] = df_new['pv_mw'] * yearly_sun_factor * cloud_noise
    df_new['irradiance'] = df_new['irradiance'] * yearly_sun_factor * cloud_noise

    # 4. 风电与风速增强 (Wind)
    # 模拟"多风年/少风年"
    yearly_wind_factor = np.random.uniform(0.90, 1.10)
    wind_gust_noise = np.random.normal(1.0, noise_level, n)

    df_new['wind_mw'] = df_new['wind_mw'] * yearly_wind_factor * wind_gust_noise
    df_new['wind_speed'] = df_new['wind_speed'] * yearly_wind_factor * wind_gust_noise

    # 5. 温度增强
    # 模拟气候变暖或冷冬: 整体偏移 ±1度
    temp_offset = np.random.uniform(-1.0, 1.0)
    # 随机波动
    temp_noise = np.random.normal(0, 0.5, n)
    df_new['temperature'] = df_new['temperature'] + temp_offset + temp_noise

    # 6. 物理截断 (清洗数据)
    # 防止负数
    cols_to_clip = ['load_mw', 'pv_mw', 'wind_mw', 'irradiance', 'wind_speed']
    for col in cols_to_clip:
        df_new[col] = df_new[col].clip(lower=0)

    return df_new


def main():
    print("1. 读取原始数据...")
    df_base = pd.read_csv(INPUT_FILE)

    # 确保原始年份是 2034
    df_base['date'] = pd.to_datetime(df_base['date'])
    start_year = df_base['date'].dt.year.min()
    print(f"   基础年份: {start_year}, 数据点: {len(df_base)}")

    frames = []

    # 生成 3 年数据 (含原始年份)
    # Year 0: 原始数据 (稍微加一点点噪声防止过拟合)
    print("2. 生成第 1 年数据 (Base)...")
    df_y1 = augment_year(df_base, year_offset=0, noise_level=0.01)
    frames.append(df_y1)

    print("3. 生成第 2 年数据 (Augmented)...")
    df_y2 = augment_year(df_base, year_offset=1, noise_level=0.03)
    frames.append(df_y2)

    print("4. 生成第 3 年数据 (Augmented)...")
    df_y3 = augment_year(df_base, year_offset=2, noise_level=0.05)
    frames.append(df_y3)

    # 合并
    df_final = pd.concat(frames, ignore_index=True)

    print(
        f"5. 最终数据集统计:\n   总行数: {len(df_final)}\n   时间跨度: {df_final['date'].min()} 到 {df_final['date'].max()}")

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"   已保存至: {OUTPUT_FILE}")

    # 可视化对比 (前 3 天 vs 一年后的前 3 天)
    plt.figure(figsize=(12, 6))
    subset_len = 96 * 3  # 3天

    plt.plot(df_y1['load_mw'].iloc[:subset_len].values, label='Year 1 Load', alpha=0.7)
    plt.plot(df_y2['load_mw'].iloc[:subset_len].values, label='Year 2 Load (Augmented)', alpha=0.7, linestyle='--')
    plt.title("Data Augmentation Verification: Year 1 vs Year 2 (First 3 Days)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()