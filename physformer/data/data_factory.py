import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


def time_features(dates, freq='h'):
    """
    Extract sin/cos time features with stable calendar handling.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)

    month = dates.month.values
    day = dates.day.values
    weekday = dates.weekday.values
    hour = dates.hour.values
    minute = dates.minute.values
    days_in_month = dates.days_in_month.values

    hour_float = hour + minute / 60.0

    month_norm = 2 * np.pi * (month - 1) / 12.0
    day_norm = 2 * np.pi * (day - 1) / days_in_month
    week_norm = 2 * np.pi * weekday / 7.0
    hour_norm = 2 * np.pi * hour_float / 24.0

    return np.stack([
        np.sin(month_norm), np.cos(month_norm),
        np.sin(day_norm), np.cos(day_norm),
        np.sin(week_norm), np.cos(week_norm),
        np.sin(hour_norm), np.cos(hour_norm),
    ], axis=1)


class VPPDataset(Dataset):
    """
    Generic forecasting dataset with config-driven target/covariate selection.
    """

    def __init__(self, root_path, data_path='vpp_dataset_3years.csv',
                 flag='train', size=None,
                 features='M', target=None, scale=True,
                 noise_level=0.03,
                 time_col='date', id_col=None, region_col=None,
                 target_cols=None, covariate_cols=None,
                 task_mode='component_multitask'):

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
        self.time_col = time_col or 'date'
        self.id_col = id_col
        self.region_col = region_col
        self.task_mode = task_mode or 'component_multitask'

        self.explicit_target_cols = list(target_cols) if target_cols else None
        self.explicit_covariate_cols = list(covariate_cols) if covariate_cols else None

        self.scaler = StandardScaler()
        self.feature_cols = []
        self.target_cols = []
        self.covariate_cols = []
        self.target_num = 0
        self.covariate_num = 0
        self.group_ids = []
        self.group_region_ids = []
        self.group_x_tensors = []
        self.group_y_tensors = []
        self.group_stamp_tensors = []
        self.group_time_indices = []
        self.sample_index = []
        self.col_ranges = {}

        self.__read_data__()

    def _resolve_columns(self, df_raw):
        reserved = {self.time_col}
        if self.id_col:
            reserved.add(self.id_col)
        if self.region_col:
            reserved.add(self.region_col)

        default_component_targets = ['load_mw', 'pv_mw', 'wind_mw']
        default_weather_covariates = ['temperature', 'irradiance', 'wind_speed']

        data_cols = [c for c in df_raw.columns if c not in reserved]

        if self.explicit_target_cols is not None:
            target_cols = self.explicit_target_cols
        elif self.task_mode == 'net_injection':
            if 'p_vpp_mw' in df_raw.columns:
                target_cols = ['p_vpp_mw']
            elif 'net_injection_mw' in df_raw.columns:
                target_cols = ['net_injection_mw']
            else:
                raise ValueError(
                    "task_mode='net_injection' requires target_cols or a built-in "
                    "target column such as 'p_vpp_mw'/'net_injection_mw'."
                )
        else:
            existing = [c for c in default_component_targets if c in df_raw.columns]
            target_cols = existing if existing else data_cols[:3]

        if self.explicit_covariate_cols is not None:
            covariate_cols = self.explicit_covariate_cols
        else:
            existing = [c for c in default_weather_covariates if c in df_raw.columns and c not in target_cols]
            covariate_cols = existing if existing else [c for c in data_cols if c not in target_cols]

        missing = [c for c in target_cols + covariate_cols if c not in df_raw.columns]
        if missing:
            raise ValueError(f"Missing configured columns in dataset: {missing}")

        feature_cols = list(dict.fromkeys(target_cols + covariate_cols))
        return target_cols, covariate_cols, feature_cols

    def _split_one_group(self, df_group):
        n = len(df_group)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test

        border1s = [0, max(0, num_train - self.seq_len), max(0, n - num_test - self.seq_len)]
        border2s = [num_train, num_train + num_vali, n]
        return border1s[self.set_type], border2s[self.set_type], border1s[0], border2s[0]

    def __read_data__(self):
        full_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(full_path)

        if self.time_col not in df_raw.columns:
            raise ValueError(f"Missing time column '{self.time_col}' in {full_path}")

        df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col])

        sort_cols = [self.time_col]
        if self.id_col and self.id_col in df_raw.columns:
            sort_cols = [self.id_col, self.time_col]
        df_raw = df_raw.sort_values(sort_cols).reset_index(drop=True)

        self.target_cols, self.covariate_cols, self.feature_cols = self._resolve_columns(df_raw)
        self.target_num = len(self.target_cols)
        self.covariate_num = len(self.covariate_cols)

        if self.id_col and self.id_col in df_raw.columns:
            grouped = [(str(group_id), df_group.copy()) for group_id, df_group in df_raw.groupby(self.id_col, sort=False)]
        else:
            grouped = [('__global__', df_raw.copy())]

        train_feature_frames = []
        prepared_groups = []

        for group_id, df_group in grouped:
            df_group = df_group.sort_values(self.time_col).reset_index(drop=True)
            border1, border2, train_border1, train_border2 = self._split_one_group(df_group)

            train_feature_frames.append(df_group.loc[train_border1:train_border2 - 1, self.feature_cols])
            prepared_groups.append((group_id, df_group, border1, border2))

        if self.scale:
            train_frame = pd.concat(train_feature_frames, axis=0, ignore_index=True)
            self.scaler.fit(train_frame.values)
        else:
            self.scaler = None

        for group_id, df_group, border1, border2 in prepared_groups:
            feature_frame = df_group[self.feature_cols]
            if self.scale:
                data = self.scaler.transform(feature_frame.values)
            else:
                data = feature_frame.values

            df_stamp = df_group[[self.time_col]].iloc[border1:border2].copy()
            data_stamp = time_features(df_stamp[self.time_col].values, freq='t')

            data_slice = data[border1:border2]
            if len(data_slice) == 0:
                continue

            self.group_ids.append(group_id)
            region_value = None
            if self.region_col and self.region_col in df_group.columns:
                region_value = df_group[self.region_col].iloc[0]
            self.group_region_ids.append(region_value)
            self.group_x_tensors.append(torch.tensor(data_slice, dtype=torch.float32))
            self.group_y_tensors.append(torch.tensor(data_slice, dtype=torch.float32))
            self.group_stamp_tensors.append(torch.tensor(data_stamp, dtype=torch.float32))
            self.group_time_indices.append(pd.to_datetime(df_stamp[self.time_col].values))

        for group_idx, group_tensor in enumerate(self.group_x_tensors):
            available = len(group_tensor) - self.seq_len - self.pred_len + 1
            if available <= 0:
                continue
            for start_idx in range(available):
                self.sample_index.append((group_idx, start_idx))

        if self.scale and self.scaler is not None:
            train_frame = pd.concat(train_feature_frames, axis=0, ignore_index=True)
            for col in train_frame.columns:
                self.col_ranges[col] = train_frame[col].max() - train_frame[col].min()

    def __getitem__(self, index):
        group_idx, s_begin = self.sample_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.group_x_tensors[group_idx][s_begin:s_end]
        seq_y = self.group_y_tensors[group_idx][r_begin:r_end]
        seq_x_mark = self.group_stamp_tensors[group_idx][s_begin:s_end]
        seq_y_mark = self.group_stamp_tensors[group_idx][r_begin:r_end]

        if self.set_type == 0 and self.noise_level > 0 and self.covariate_num > 0:
            seq_x = seq_x.clone()
            noise = torch.randn_like(seq_x) * self.noise_level
            seq_x[:, self.target_num:] += noise[:, self.target_num:]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_index)

    def inverse_transform(self, data):
        if not self.scale or self.scaler is None:
            return data

        if torch.is_tensor(data):
            data = data.cpu().numpy()

        original_shape = data.shape
        if len(original_shape) == 3:
            data = data.reshape(-1, original_shape[-1])

        n_expected = self.scaler.n_features_in_
        n_input = data.shape[1]

        if n_input == n_expected:
            data_rescaled = self.scaler.inverse_transform(data)
        elif n_input < n_expected:
            dummy = np.zeros((data.shape[0], n_expected))
            dummy[:, :n_input] = data
            data_rescaled = self.scaler.inverse_transform(dummy)[:, :n_input]
        else:
            raise ValueError(f"Input features ({n_input}) > scaler features ({n_expected})")

        if len(original_shape) == 3:
            data_rescaled = data_rescaled.reshape(original_shape)

        return data_rescaled

    def get_prediction_metadata(self, index):
        group_idx, s_begin = self.sample_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        times = self.group_time_indices[group_idx][r_end - self.pred_len:r_end]
        return {
            'portfolio_id': self.group_ids[group_idx],
            'region_id': self.group_region_ids[group_idx],
            'forecast_timestamps': pd.to_datetime(times),
        }

    def get_schema(self):
        return {
            'task_mode': self.task_mode,
            'time_col': self.time_col,
            'id_col': self.id_col,
            'region_col': self.region_col,
            'target_cols': self.target_cols.copy(),
            'covariate_cols': self.covariate_cols.copy(),
            'feature_cols': self.feature_cols.copy(),
        }


class PhysFormerDataset(VPPDataset):
    """
    PhysFormer-specific dataset.

    Supported only for component_multitask with exactly:
    - 3 targets
    - 3 weather covariates
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.task_mode != 'component_multitask':
            raise ValueError(
                "PhysFormerDataset currently supports only task_mode='component_multitask'. "
                "Use the generic VPPDataset for net injection forecasting until the model "
                "is generalized beyond the fixed 3-target architecture."
            )
        if self.target_num != 3 or self.covariate_num != 3:
            raise ValueError(
                "PhysFormerDataset requires exactly 3 target columns and 3 covariate columns. "
                f"Got targets={self.target_cols}, covariates={self.covariate_cols}."
            )

        self._compute_weather_stats()

    def _compute_weather_stats(self):
        if self.scale and self.scaler is not None:
            weather_start = self.target_num
            weather_indices = list(range(weather_start, weather_start + self.covariate_num))
            self.weather_mean = self.scaler.mean_[weather_indices]
            self.weather_std = self.scaler.scale_[weather_indices]
        else:
            self.weather_mean = np.zeros(self.covariate_num)
            self.weather_std = np.ones(self.covariate_num)

    def get_scaler_params(self):
        if self.scale and self.scaler is not None:
            return {
                'mean': self.scaler.mean_.copy(),
                'std': self.scaler.scale_.copy(),
                'weather_mean': self.weather_mean.copy(),
                'weather_std': self.weather_std.copy(),
            }
        return {
            'mean': None,
            'std': None,
            'weather_mean': None,
            'weather_std': None,
        }

    def get_physical_stats(self):
        stats = {
            'means': None,
            'stds': None,
            'ramp_limits': None,
        }

        if self.scale and self.scaler is not None:
            stats['means'] = self.scaler.mean_[:self.target_num]
            stats['stds'] = self.scaler.scale_[:self.target_num]

        try:
            raw_groups = []
            for group_tensor in self.group_x_tensors:
                group_np = group_tensor.cpu().numpy()
                raw_groups.append(self.inverse_transform(group_np))
            if raw_groups:
                raw_data = np.concatenate(raw_groups, axis=0)
                target_data = raw_data[:, :self.target_num]
                diff = np.abs(target_data[1:] - target_data[:-1])
                stats['ramp_limits'] = np.percentile(diff, 99.9, axis=0) * 1.5
        except Exception as exc:
            print(f"[Dataset Warning] Failed to calculate ramp limits: {exc}")
            stats['ramp_limits'] = None

        return stats

    def __getitem__(self, index):
        group_idx, s_begin = self.sample_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_raw = self.group_x_tensors[group_idx][s_begin:s_end]
        seq_y_raw = self.group_y_tensors[group_idx][r_begin:r_end]

        x_stat = seq_raw[:, :self.target_num]
        x_weather_hist = seq_raw[:, self.target_num:self.target_num + self.covariate_num]
        x_weather_future = seq_y_raw[-self.pred_len:, self.target_num:self.target_num + self.covariate_num]
        y = seq_y_raw[-self.pred_len:, :self.target_num]

        x_mark_enc = self.group_stamp_tensors[group_idx][s_begin:s_end]
        y_mark = self.group_stamp_tensors[group_idx][r_begin:r_end]

        if self.set_type == 0 and self.noise_level > 0:
            x_weather_hist = x_weather_hist.clone()
            x_weather_future = x_weather_future.clone()
            x_weather_hist += torch.randn_like(x_weather_hist) * self.noise_level
            x_weather_future += torch.randn_like(x_weather_future) * self.noise_level

        return x_stat, x_weather_hist, x_weather_future, y, x_mark_enc, y_mark


def data_provider(args, flag):
    if args.model == 'PhysFormer':
        Data = PhysFormerDataset
    else:
        Data = VPPDataset

    shuffle_flag = (flag == 'train')
    drop_last = (flag == 'train')
    batch_size = args.batch_size if flag != 'test' else 1

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        noise_level=0.03 if flag == 'train' else 0.0,
        time_col=getattr(args, 'time_col', 'date'),
        id_col=getattr(args, 'id_col', None),
        region_col=getattr(args, 'region_col', None),
        target_cols=getattr(args, 'target_cols', None),
        covariate_cols=getattr(args, 'covariate_cols', None),
        task_mode=getattr(args, 'task_mode', 'component_multitask'),
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    return data_set, data_loader
