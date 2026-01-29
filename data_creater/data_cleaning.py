import pandas as pd
import numpy as np


def clean_and_shift_time():
    input_file = '../data/vpp_training_data.csv'
    output_file = '../data/vpp_training_data(cleaned).csv'

    print(f"1. 读取文件: {input_file} ...")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # --- 步骤 A: 基础清洗 (去除重复) ---
    print(f"   原始数据量: {len(df)}")
    df = df.drop_duplicates(subset=['date'], keep='first')
    df = df.sort_values('date').reset_index(drop=True)

    # --- 步骤 B: 修复原始时间轴断点 (插值) ---
    # 先基于原始时间生成完美索引，填补空洞（如原数据缺少的 2月29日）
    start_orig = df['date'].min()
    end_orig = df['date'].max()
    full_idx_orig = pd.date_range(start=start_orig, end=end_orig, freq='15min')

    # 重建索引并插值
    df = df.set_index('date').reindex(full_idx_orig)

    # 数值列线性插值 (保持物理平滑)
    cols_float = df.select_dtypes(include=['float64']).columns
    df[cols_float] = df[cols_float].interpolate(method='linear')

    # 标志位列 (Flag) 向前填充 (保持离散性质)
    # 假设 violation_flag 是 int 或 float，也用 ffill 防止出现 0.5 这种值
    if 'violation_flag' in df.columns:
        df['violation_flag'] = df['violation_flag'].ffill().bfill()

    df = df.reset_index(drop=True)  # 丢弃旧的时间索引

    # --- 步骤 C: 生成全新的 2023 时间轴 ---
    print("2. 执行时间平移 (Shift to 2023)...")
    # 生成与数据长度完全一致的新时间轴
    new_start_date = "2023-01-01 00:00:00"
    new_time_index = pd.date_range(start=new_start_date, periods=len(df), freq='15min')

    # 覆盖旧时间
    df['date'] = new_time_index

    # --- 步骤 D: 验证与保存 ---
    print(f"   新起止时间: {df['date'].min()} -> {df['date'].max()}")

    # 检查 2024 闰年是否完整
    leap_day_count = len(df[df['date'].dt.date.astype(str) == '2024-02-29'])
    print(f"   2024-02-29 数据点数: {leap_day_count} (应为 96)")

    df.to_csv(output_file, index=False)
    print(f"\n成功！已保存至: {output_file}")
    print("这份数据现在：\n 1. 没有任何断点\n 2. 包含完整的闰年数据\n 3. 从 2023-01-01 开始")


if __name__ == "__main__":
    clean_and_shift_time()