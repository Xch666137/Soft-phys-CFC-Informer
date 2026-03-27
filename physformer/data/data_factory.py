import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

_PREPARED_DATA_CACHE = {}
_TARGET_COLUMNS = ['Load', 'PV', 'Wind']
_WEATHER_COLUMNS = ['Temp', 'Irradiance', 'WindSpeed']
_EXPECTED_VPP_SCHEMA = ['date', *_TARGET_COLUMNS, *_WEATHER_COLUMNS]


def time_features(dates, freq='h'):
    """
    Manual cyclic time encodings.

    The encoding itself is frequency-agnostic, but the caller now decides which
    frequency is semantically expected so the dataset and model stay aligned.
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


def _data_cache_key(root_path, data_path, features, target, scale, freq):
    return (
        os.path.abspath(os.path.join(root_path, data_path)),
        features,
        target,
        bool(scale),
        freq,
    )


def _validate_dataframe(df_raw, features, target):
    if 'date' not in df_raw.columns:
        raise ValueError("Dataset must contain a 'date' column.")

    if features in ['M', 'MS']:
        required = _TARGET_COLUMNS + _WEATHER_COLUMNS
        missing = [col for col in required if col not in df_raw.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns for multivariate mode: {missing}")
        actual_prefix = df_raw.columns[:len(_EXPECTED_VPP_SCHEMA)].tolist()
        if actual_prefix != _EXPECTED_VPP_SCHEMA:
            raise ValueError(
                "Dataset schema mismatch for multivariate mode. "
                f"Expected leading columns {_EXPECTED_VPP_SCHEMA}, got {actual_prefix}"
            )
    elif features == 'S':
        if not target:
            raise ValueError("Single-target mode requires a non-empty target column.")
        if target not in df_raw.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")
    else:
        raise ValueError(f"Unsupported features mode: {features}")


def _build_prepared_data(root_path, data_path, features, target, scale, freq):
    key = _data_cache_key(root_path, data_path, features, target, scale, freq)
    cached = _PREPARED_DATA_CACHE.get(key)
    if cached is not None:
        return cached

    csv_path = os.path.abspath(os.path.join(root_path, data_path))
    df_raw = pd.read_csv(csv_path)
    _validate_dataframe(df_raw, features, target)

    if features in ['M', 'MS']:
        feature_columns = _TARGET_COLUMNS + _WEATHER_COLUMNS
        df_data = df_raw[feature_columns]
    else:
        feature_columns = [target]
        df_data = df_raw[feature_columns]

    scaler = StandardScaler() if scale else None
    if scale:
        num_train = int(len(df_raw) * 0.7)
        scaler.fit(df_data.iloc[:num_train].values)
        data_values = scaler.transform(df_data.values)
    else:
        data_values = df_data.values

    prepared = {
        'df_raw': df_raw,
        'feature_columns': feature_columns,
        'data_values': data_values,
        'date_features': time_features(df_raw['date'].values, freq=freq),
        'scaler': scaler,
        'num_train': int(len(df_raw) * 0.7),
        'num_test': int(len(df_raw) * 0.2),
    }
    prepared['num_val'] = len(df_raw) - prepared['num_train'] - prepared['num_test']

    if scale:
        raw_train = df_data.iloc[:prepared['num_train']]
        prepared['col_ranges'] = {
            col: raw_train[col].max() - raw_train[col].min()
            for col in raw_train.columns
        }
    else:
        prepared['col_ranges'] = {}

    _PREPARED_DATA_CACHE[key] = prepared
    return prepared


class VPPDataset(Dataset):
    """Base VPP dataset."""

    def __init__(self, root_path, data_path='vpp_dataset_3years.csv',
                 flag='train', size=None, features='M', target=None, scale=True,
                 noise_level=0.03, freq='t'):

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
        self.flag = flag
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.noise_level = noise_level
        self.root_path = root_path
        self.data_path = data_path
        self.freq = freq

        self.__read_data__()

    def __read_data__(self):
        prepared = _build_prepared_data(
            root_path=self.root_path,
            data_path=self.data_path,
            features=self.features,
            target=self.target,
            scale=self.scale,
            freq=self.freq,
        )

        self.scaler = prepared['scaler']
        self.feature_columns = prepared['feature_columns']
        self.col_ranges = prepared['col_ranges']

        num_train = prepared['num_train']
        num_test = prepared['num_test']
        num_val = prepared['num_val']

        border1s = [0, num_train - self.seq_len, len(prepared['df_raw']) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(prepared['df_raw'])]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = prepared['data_values'][border1:border2]
        self.data_y = prepared['data_values'][border1:border2]
        self.data_stamp = prepared['date_features'][border1:border2]

        self.target_num = min(3, self.data_x.shape[1])
        self.covariate_num = max(0, self.data_x.shape[1] - self.target_num)

        self.data_x_tensor = torch.tensor(self.data_x, dtype=torch.float32)
        self.data_y_tensor = torch.tensor(self.data_y, dtype=torch.float32)
        self.data_stamp_tensor = torch.tensor(self.data_stamp, dtype=torch.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x_tensor[s_begin:s_end]
        seq_y = self.data_y_tensor[r_begin:r_end]
        seq_x_mark = self.data_stamp_tensor[s_begin:s_end]
        seq_y_mark = self.data_stamp_tensor[r_begin:r_end]

        if self.set_type == 0 and self.noise_level > 0:
            seq_x = seq_x.clone()
            noise = torch.randn_like(seq_x) * self.noise_level
            if seq_x.shape[1] > self.target_num:
                seq_x[:, self.target_num:] += noise[:, self.target_num:]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse transform with support for partial feature slices."""
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
                f"Input features ({n_features_input}) > Scaler features ({n_features_expected})"
            )

        if len(original_shape) == 3:
            data_rescaled = data_rescaled.reshape(original_shape)

        return data_rescaled


class PhysFormerDataset(VPPDataset):
    """
    PhysFormer-specific dataset.

    Expected CSV column order is still:
    [date, Load, PV, Wind, Temp, Irradiance, WindSpeed]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_weather_stats()

    def _compute_weather_stats(self):
        if self.scale and self.scaler is not None and self.scaler.mean_.shape[0] >= 6:
            weather_indices = [3, 4, 5]
            self.weather_mean = self.scaler.mean_[weather_indices]
            self.weather_std = self.scaler.scale_[weather_indices]
        else:
            self.weather_mean = np.array([20.0, 400.0, 5.0])
            self.weather_std = np.array([10.0, 300.0, 3.0])

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
            'weather_mean': self.weather_mean.copy(),
            'weather_std': self.weather_std.copy(),
        }

    def get_physical_stats(self):
        stats = {'means': None, 'stds': None, 'ramp_limits': None}

        if self.scale and self.scaler is not None:
            stats['means'] = self.scaler.mean_[:3]
            stats['stds'] = self.scaler.scale_[:3]

        try:
            raw_data = self.inverse_transform(self.data_x) if self.scale else self.data_x
            target_data = raw_data[:, :3]
            diff = np.abs(target_data[1:] - target_data[:-1])
            stats['ramp_limits'] = np.percentile(diff, 99.9, axis=0) * 1.5
        except Exception as exc:
            print(f"[Dataset Warning] Failed to calculate ramp limits: {exc}")
            stats['ramp_limits'] = None

        return stats

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_raw = self.data_x_tensor[s_begin:s_end]
        seq_y_raw = self.data_y_tensor[r_begin:r_end]

        x_stat = seq_raw[:, 0:3]
        x_weather_hist = seq_raw[:, 3:6]
        x_weather_future = seq_y_raw[-self.pred_len:, 3:6]
        y = seq_y_raw[-self.pred_len:, 0:3]

        x_mark_enc = self.data_stamp_tensor[s_begin:s_end]
        y_mark = self.data_stamp_tensor[r_begin:r_end]

        if self.set_type == 0 and self.noise_level > 0:
            x_weather_hist = x_weather_hist.clone()
            x_weather_future = x_weather_future.clone()
            x_weather_hist += torch.randn_like(x_weather_hist) * self.noise_level
            x_weather_future += torch.randn_like(x_weather_future) * self.noise_level

        return x_stat, x_weather_hist, x_weather_future, y, x_mark_enc, y_mark


def _resolve_dataset_class(args):
    trainer_family = getattr(args, 'trainer_family', None)
    if trainer_family == 'physformer' or getattr(args, 'model', None) == 'PhysFormer':
        return PhysFormerDataset
    return VPPDataset


def _resolve_batch_size(args, flag, dataset_class):
    if flag == 'train':
        return getattr(args, 'batch_size')
    if flag == 'val':
        return getattr(args, 'val_batch_size', getattr(args, 'batch_size'))
    explicit_test_batch = getattr(args, 'test_batch_size', None)
    if explicit_test_batch is not None:
        return explicit_test_batch
    return 1 if dataset_class is VPPDataset else getattr(args, 'batch_size')


def data_provider(args, flag):
    """
    Shared dataset/dataloader factory for PhysFormer and baselines.
    """
    dataset_class = _resolve_dataset_class(args)
    shuffle_flag = flag == 'train'
    drop_last = flag == 'train'
    batch_size = _resolve_batch_size(args, flag, dataset_class)
    train_noise_level = getattr(args, 'train_noise_level', 0.0)

    data_set = dataset_class(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        noise_level=train_noise_level if flag == 'train' else 0.0,
        freq=getattr(args, 'freq', 't'),
    )

    num_workers = getattr(args, 'num_workers', 0)
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle_flag,
        'num_workers': num_workers,
        'drop_last': drop_last,
        'pin_memory': bool(getattr(args, 'pin_memory', getattr(args, 'use_gpu', False))),
    }

    if num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(
            getattr(args, 'persistent_workers', True)
        )
        prefetch_factor = getattr(args, 'prefetch_factor', None)
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor

    data_loader = DataLoader(data_set, **loader_kwargs)
    return data_set, data_loader
