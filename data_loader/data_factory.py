import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


# --- 新增：时间特征编码逻辑 ---
def time_features(dates, freq='h'):
    """
    手动提取时间特征并进行 Sin/Cos 编码 (8维)
    [Month, Day, Weekday, Hour] * [Sin, Cos]
    """
    # 转换为 Pandas DateTime
    dates = pd.to_datetime(dates)

    # 提取基础特征
    month = dates.month.values
    day = dates.day.values
    weekday = dates.weekday.values
    hour = dates.hour.values
    minute = dates.minute.values

    # 将 Hour 优化为 Hour + Minute/60，以保留 15min 的细粒度
    hour_float = hour + minute / 60.0

    # 定义周期
    # Month: 1-12
    # Day: 1-31
    # Weekday: 0-6
    # Hour: 0-24

    # 归一化并编码 (Sin, Cos)
    # Month
    month_norm = 2 * np.pi * (month - 1) / 11.0
    month_sin = np.sin(month_norm)
    month_cos = np.cos(month_norm)

    # Day
    day_norm = 2 * np.pi * (day - 1) / 30.0
    day_sin = np.sin(day_norm)
    day_cos = np.cos(day_norm)

    # Weekday
    week_norm = 2 * np.pi * weekday / 6.0
    week_sin = np.sin(week_norm)
    week_cos = np.cos(week_norm)

    # Hour (Continuous)
    hour_norm = 2 * np.pi * hour_float / 24.0
    hour_sin = np.sin(hour_norm)
    hour_cos = np.cos(hour_norm)

    # 堆叠为 [L, 8]
    # 顺序: Month_Sin, Month_Cos, Day_Sin, Day_Cos, Week_Sin, Week_Cos, Hour_Sin, Hour_Cos
    dt_enc = np.stack([
        month_sin, month_cos,
        day_sin, day_cos,
        week_sin, week_cos,
        hour_sin, hour_cos
    ], axis=1)

    return dt_enc


class VPPDataset(Dataset):
    def __init__(self, root_path, data_path='vpp_dataset_3years.csv',
                 flag='train', size=None,
                 features='M', target=None, scale=True,
                 noise_level=0.03):
        """
        Args:
            root_path: 数据文件所在目录
            data_path: 数据文件名
            flag: 'train', 'val', 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M' (多变量预测), 'S' (单变量), 'MS' (多变量预测单变量)
            target: 预测目标列名 (仅在 features='S' 时生效)
            scale: 是否启用全局归一化 (StandardScaler)
            noise_level: 噪声注入强度 (仅在训练集生效)
        """
        # 1. 参数初始化
        if size is None:
            self.seq_len = 96 * 4 * 7  # 默认 7 天
            self.label_len = 48  # 默认 12 小时
            self.pred_len = 96  # 默认 24 小时
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val'], "flag must be one of ['train', 'test', 'val']"

        # 划分策略：70% 训练, 10% 验证, 20% 测试
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.noise_level = noise_level

        self.root_path = root_path
        self.data_path = data_path

        # 2. 读取与处理数据
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 划分数据集索引
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 特征选择逻辑
        if self.features == 'M' or self.features == 'MS':
            # 多变量预测：使用所有列 (跳过第一列 date)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 单变量预测：只使用 target 列
            df_data = df_raw[[self.target]]

        # --- 归一化逻辑 ---
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # --- 时间特征编码 (关键修复) ---
        # 读取第一列 'date'
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # 调用自定义函数生成 8 维特征
        data_stamp = time_features(df_stamp['date'].values, freq='t')

        # 记录每列的数据范围 (用于 NRMSE 计算)
        self.col_ranges = {}
        if self.scale:
            raw_train = df_data.iloc[border1s[0]:border2s[0]]
            for col in raw_train.columns:
                self.col_ranges[col] = raw_train[col].max() - raw_train[col].min()
        else:
            train_slice = df_data.iloc[border1s[0]:border2s[0]]
            for col in train_slice.columns:
                self.col_ranges[col] = train_slice[col].max() - train_slice[col].min()

        # 切分当前 split 的数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp  # 保存时间特征

        # 记录协变量数量
        self.target_num = 3
        self.covariate_num = data.shape[1] - self.target_num

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        # Decoder Input 的切片逻辑
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # --- 使用真正的 8 维时间特征 ---
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 噪声注入 (仅针对训练集)
        if self.set_type == 0 and self.noise_level > 0:
            seq_x = seq_x.copy()
            noise = np.random.normal(0, self.noise_level, seq_x.shape)
            if seq_x.shape[1] > self.target_num:
                seq_x[:, self.target_num:] += noise[:, self.target_num:]

        return torch.tensor(seq_x, dtype=torch.float32), \
            torch.tensor(seq_y, dtype=torch.float32), \
            torch.tensor(seq_x_mark, dtype=torch.float32), \
            torch.tensor(seq_y_mark, dtype=torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        增强版反归一化：支持 2D/3D 输入，支持部分列反归一化（自动补全）
        """
        if not self.scale or self.scaler is None:
            return data

        # 1. 统一转为 Numpy
        if torch.is_tensor(data):
            data = data.cpu().numpy()

        # 2. 形状处理 (记录原始形状)
        original_shape = data.shape
        if len(data.shape) == 3:
            # [Batch, Seq, Feat] -> [Batch*Seq, Feat]
            data = data.reshape(-1, data.shape[-1])

        # 3. 维度匹配检查与补全
        n_features_expected = self.scaler.n_features_in_
        n_features_input = data.shape[1]

        if n_features_input == n_features_expected:
            data_rescaled = self.scaler.inverse_transform(data)
        elif n_features_input < n_features_expected:
            dummy = np.zeros((data.shape[0], n_features_expected))
            dummy[:, :n_features_input] = data
            dummy_rescaled = self.scaler.inverse_transform(dummy)
            data_rescaled = dummy_rescaled[:, :n_features_input]
        else:
            raise ValueError(
                f"Input features ({n_features_input}) > Scaler features ({n_features_expected}). Check data definition.")

        # 4. 恢复 3D 形状
        if len(original_shape) == 3:
            data_rescaled = data_rescaled.reshape(original_shape[0], original_shape[1], original_shape[2])

        return data_rescaled


# data_loader/data_factory.py 中的新增代码

class PhysFormerDataset(VPPDataset):
    """
    PhysFormer 专用数据集
    输出解耦后的双流数据：
    1. x_stat (统计流): [Load, PV, Wind] -> 关注历史惯性
    2. x_phys (物理流): [Temp, Irr, Speed, ΔLoad, ΔPV, ΔWind] -> 关注驱动力与变化率
    """

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 1. 原始切片 [Seq, 6]
        # (假设列顺序: 0:Load, 1:PV, 2:Wind, 3:Temp, 4:Irr, 5:Speed)
        seq_raw = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 2. 构建统计流输入 (Stat Stream) -> [Seq, 3]
        # 只取前3列功率数据
        seq_stat = seq_raw[:, 0:3]

        # 3. 构建物理流输入 (Phys Stream) -> [Seq, 6]
        # Part A: 气象驱动 (Temp, Irr, Speed)
        seq_weather = seq_raw[:, 3:6]

        # Part B: 动力学差分 (ΔPower)
        # 注意：这里需要往回多取一个点来计算第一个时刻的差分，或者简单地补0
        # 简单处理：内部计算差分，第一位补0
        seq_power = seq_raw[:, 0:3]
        seq_diff = np.zeros_like(seq_power)
        seq_diff[1:, :] = seq_power[1:, :] - seq_power[:-1, :]
        seq_diff[0, :] = 0  # 保持第一帧稳定

        # 物理流拼接: [Weather(3), Diff(3)]
        seq_phys = np.concatenate([seq_weather, seq_diff], axis=1)

        # 4. 定向噪声注入 (Physics Augmentation) - 仅在训练时
        if self.set_type == 0 and self.noise_level > 0:
            # 方案：只给“天气预测”加噪声，模拟气象预报的不确定性
            # 不给“历史功率”加噪声，因为那是已知事实
            noise = np.random.normal(0, self.noise_level, seq_weather.shape)
            seq_phys[:, 0:3] += noise

            # 返回修改后的 Tuple结构: (stat, phys, y, x_mark, y_mark)
        return (torch.tensor(seq_stat, dtype=torch.float32),
                torch.tensor(seq_phys, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(seq_x_mark, dtype=torch.float32),
                torch.tensor(seq_y_mark, dtype=torch.float32))