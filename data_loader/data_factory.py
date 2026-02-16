import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


# --- 时间特征编码逻辑 ---
def time_features(dates, freq='h'):
    """
    手动提取时间特征并进行 Sin/Cos 编码 (8维)
    [Month, Day, Weekday, Hour] * [Sin, Cos]
    """
    dates = pd.to_datetime(dates)

    month = dates.month.values
    day = dates.day.values
    weekday = dates.weekday.values
    hour = dates.hour.values
    minute = dates.minute.values

    hour_float = hour + minute / 60.0

    # 归一化并编码
    month_norm = 2 * np.pi * (month - 1) / 11.0
    month_sin = np.sin(month_norm)
    month_cos = np.cos(month_norm)

    day_norm = 2 * np.pi * (day - 1) / 30.0
    day_sin = np.sin(day_norm)
    day_cos = np.cos(day_norm)

    week_norm = 2 * np.pi * weekday / 6.0
    week_sin = np.sin(week_norm)
    week_cos = np.cos(week_norm)

    hour_norm = 2 * np.pi * hour_float / 24.0
    hour_sin = np.sin(hour_norm)
    hour_cos = np.cos(hour_norm)

    dt_enc = np.stack([
        month_sin, month_cos,
        day_sin, day_cos,
        week_sin, week_cos,
        hour_sin, hour_cos
    ], axis=1)

    return dt_enc


class VPPDataset(Dataset):
    """
    基础VPP数据集（保持向后兼容）
    """

    def __init__(self, root_path, data_path='vpp_dataset_3years.csv',
                 flag='train', size=None,
                 features='M', target=None, scale=True,
                 noise_level=0.03):

        if size is None:
            self.seq_len = 96 * 4 * 7
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.noise_level = noise_level
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 划分数据集
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 特征选择
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 归一化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 时间特征
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp['date'].values, freq='t')

        # 记录数据范围
        self.col_ranges = {}
        if self.scale:
            raw_train = df_data.iloc[border1s[0]:border2s[0]]
            for col in raw_train.columns:
                self.col_ranges[col] = raw_train[col].max() - raw_train[col].min()

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.target_num = 3
        self.covariate_num = data.shape[1] - self.target_num

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

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
        """增强版反归一化"""
        if not self.scale or self.scaler is None:
            return data

        if torch.is_tensor(data):
            data = data.cpu().numpy()

        original_shape = data.shape
        if len(data.shape) == 3:
            data = data.reshape(-1, data.shape[-1])

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
                f"Input features ({n_features_input}) > Scaler features ({n_features_expected})")

        if len(original_shape) == 3:
            data_rescaled = data_rescaled.reshape(original_shape)

        return data_rescaled


class PhysFormerDataset(VPPDataset):
    """
    PhysFormer 专用数据集

    数据格式假设：CSV列顺序为 [date, Load, PV, Wind, Temp, Irradiance, WindSpeed]

    返回格式：
        x_stat: [Seq, 3] - 历史观测 [Load, PV, Wind]
        x_weather_hist: [Seq, 3] - 历史天气 [Temp, Irr, Speed]
        x_weather_future: [Pred, 3] - 未来天气预报 [Temp, Irr, Speed]
        y: [Label+Pred, 3] - 真实标签 [Load, PV, Wind]
        x_mark_enc: [Seq, 8] - 历史时间特征
        y_mark: [Label+Pred, 8] - 未来时间特征
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_weather_stats()

    def _compute_weather_stats(self):
        """从 Scaler 中提取天气变量的统计量"""
        if self.scale and self.scaler is not None:
            # 假设数据列顺序: [Load, PV, Wind, Temp, Irr, Speed]
            # 天气特征通常在索引 3, 4, 5
            weather_indices = [3, 4, 5]

            # 确保索引不越界
            if self.scaler.mean_.shape[0] > max(weather_indices):
                self.weather_mean = self.scaler.mean_[weather_indices]
                self.weather_std = self.scaler.scale_[weather_indices]
            else:
                # 兜底：如果特征数不够（例如单变量预测），使用默认值或全0
                self.weather_mean = np.zeros(3)
                self.weather_std = np.ones(3)
        else:
            # 未归一化时的默认值（仅供参考）
            self.weather_mean = np.array([20.0, 400.0, 5.0])
            self.weather_std = np.array([10.0, 300.0, 3.0])

    def get_scaler_params(self):
        """
        Returns:
            dict: {'mean': array, 'std': array, 'weather_mean': array, 'weather_std': array}
        """
        if self.scale and self.scaler is not None:
            return {
                'mean': self.scaler.mean_.copy(),
                'std': self.scaler.scale_.copy(),
                'weather_mean': self.weather_mean.copy(),
                'weather_std': self.weather_std.copy()
            }

    def get_physical_stats(self):
        """
        [新增方法] 计算并返回物理Loss所需的所有统计量：
        1. 真实均值 (Means)
        2. 真实标准差 (Stds)
        3. 物理爬坡极限 (Ramp Limits)
        """
        stats = {
            'means': None,
            'stds': None,
            'ramp_limits': None
        }

        # 1. 提取均值和方差
        if self.scale and self.scaler is not None:
            # 取前3列: Load, PV, Wind
            stats['means'] = self.scaler.mean_[:3]
            stats['stds'] = self.scaler.scale_[:3]

        # 2. 计算爬坡极限 (Ramp Limits)
        try:
            # 获取全量数据并反归一化 (还原为物理量 MW)
            if self.scale:
                # 注意：这里直接对 self.data_x 操作，它是 numpy array
                # inverse_transform 会处理 tensor/numpy 转换
                raw_data = self.inverse_transform(self.data_x)
            else:
                raw_data = self.data_x

            # 只取前3列目标变量
            target_data = raw_data[:, :3]

            # 计算一阶差分绝对值 |t - (t-1)|
            diff = np.abs(target_data[1:] - target_data[:-1])

            # 计算 99.9% 分位数并给予 1.5 倍宽容度
            ramp_limits = np.percentile(diff, 99.9, axis=0) * 1.5

            stats['ramp_limits'] = ramp_limits

        except Exception as e:
            print(f"[Dataset Warning] Failed to calculate ramp limits: {e}")
            stats['ramp_limits'] = None

        return stats

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 1. 原始切片
        seq_raw = self.data_x[s_begin:s_end]  # [Seq, 6]
        seq_y_raw = self.data_y[r_begin:r_end]  # [Label + Pred, 6]

        # 2. 提取输入流
        x_stat = seq_raw[:, 0:3].copy()  # [Seq, 3]
        x_weather_hist = seq_raw[:, 3:6].copy()  # [Seq, 3]

        # 3. 未来数据处理
        # PhysFormer 是 Encoder-Only 模型，直接预测 'pred_len'。
        # 输出不需要包含 'label_len'，否则维度会无法对齐。
        # 我们严格切取最后 'pred_len' 个数据。
        x_weather_future = seq_y_raw[-self.pred_len:, 3:6].copy()  # [Pred, 3]
        y = seq_y_raw[-self.pred_len:, 0:3].copy()  # [Pred, 3]

        # 4. 时间特征
        x_mark_enc = self.data_stamp[s_begin:s_end].copy()
        # y_mark 在 Encoder-Only 中通常不用，但保留以兼容接口
        y_mark = self.data_stamp[r_begin:r_end].copy()

        # 5. 噪声注入 (仅训练)
        if self.set_type == 0 and self.noise_level > 0:
            noise_hist = np.random.normal(0, self.noise_level, x_weather_hist.shape)
            noise_future = np.random.normal(0, self.noise_level, x_weather_future.shape)
            x_weather_hist += noise_hist
            x_weather_future += noise_future

        # ===== Step 5: 转为Tensor并返回 =====
        return (
            torch.tensor(x_stat, dtype=torch.float32),  # [Seq, 3]
            torch.tensor(x_weather_hist, dtype=torch.float32),  # [Seq, 3]
            torch.tensor(x_weather_future, dtype=torch.float32),  # [Pred, 3]
            torch.tensor(y, dtype=torch.float32),  # [Pred, 3]
            torch.tensor(x_mark_enc, dtype=torch.float32),  # [Seq, 8]
            torch.tensor(y_mark, dtype=torch.float32)  # [Pred, 8]
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


# ===== DataLoader工厂函数 =====
def data_provider(args, flag):
    """
    数据加载器工厂函数

    Args:
        args: 参数对象，需包含以下字段：
            - root_path: 数据根目录
            - data_path: 数据文件名
            - features: 特征类型 ('M', 'S', 'MS')
            - target: 目标列名（仅S模式需要）
            - seq_len, label_len, pred_len: 序列长度
            - batch_size: 批大小
            - num_workers: 数据加载线程数
        flag: 'train', 'val', 'test'

    Returns:
        data_set: Dataset对象
        data_loader: DataLoader对象
    """
    # 选择数据集类型
    if args.model == 'PhysFormer':
        Data = PhysFormerDataset
    else:
        Data = VPPDataset

    # 确定是否打乱数据
    shuffle_flag = (flag == 'train')
    drop_last = (flag == 'train')

    # 批大小调整
    batch_size = args.batch_size
    if flag == 'test':
        batch_size = 1  # 测试时使用单样本，便于逐样本分析

    # 创建数据集
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,  # 始终启用归一化
        noise_level=0.03 if flag == 'train' else 0.0  # 仅训练时加噪声
    )

    # 创建DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
