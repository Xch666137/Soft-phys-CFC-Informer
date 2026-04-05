import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def time_features(dates, freq="h"):
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

    return np.stack(
        [
            np.sin(month_norm),
            np.cos(month_norm),
            np.sin(day_norm),
            np.cos(day_norm),
            np.sin(week_norm),
            np.cos(week_norm),
            np.sin(hour_norm),
            np.cos(hour_norm),
        ],
        axis=1,
    )


class VPPDataset(Dataset):
    """
    Generic VPP forecasting dataset with explicit column roles.
    """

    def __init__(
        self,
        root_path,
        data_path="vpp_dataset_3years.csv",
        flag="train",
        size=None,
        features="M",
        target=None,
        scale=True,
        noise_level=0.03,
        time_col="date",
        id_col=None,
        region_col=None,
        split_col=None,
        split_strategy="time_series",
        target_cols=None,
        covariate_cols=None,
        known_future_covariate_cols=None,
        history_state_cols=None,
        aux_target_cols=None,
        task_mode="component_multitask",
    ):
        if size is None:
            self.seq_len = 96 * 4 * 7
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.noise_level = noise_level
        self.root_path = root_path
        self.data_path = data_path
        self.time_col = time_col or "date"
        self.id_col = id_col
        self.region_col = region_col
        self.split_col = split_col
        self.split_strategy = split_strategy or "time_series"
        self.task_mode = task_mode or "component_multitask"

        self.explicit_target_cols = list(target_cols) if target_cols else None
        self.explicit_known_future_covariate_cols = (
            list(known_future_covariate_cols) if known_future_covariate_cols else None
        )
        self.explicit_history_state_cols = list(history_state_cols) if history_state_cols else []
        self.explicit_aux_target_cols = list(aux_target_cols) if aux_target_cols else []
        self.legacy_covariate_cols = list(covariate_cols) if covariate_cols else None

        self.feature_scaler = StandardScaler()
        self.aux_scaler = StandardScaler()

        self.target_cols = []
        self.known_future_covariate_cols = []
        self.history_state_cols = []
        self.aux_target_cols = []
        self.feature_cols = []
        self.target_num = 0
        self.known_future_num = 0
        self.history_state_num = 0
        self.aux_target_num = 0
        self.col_ranges = {}

        self.group_ids = []
        self.group_region_ids = []
        self.group_x_tensors = []
        self.group_y_tensors = []
        self.group_aux_tensors = []
        self.group_stamp_tensors = []
        self.group_time_indices = []
        self.sample_index = []

        self._train_feature_frame = None
        self._train_aux_frame = None

        self.__read_data__()

    def _resolve_columns(self, df_raw):
        reserved = {self.time_col}
        if self.id_col:
            reserved.add(self.id_col)
        if self.region_col:
            reserved.add(self.region_col)
        if self.split_col:
            reserved.add(self.split_col)

        data_cols = [c for c in df_raw.columns if c not in reserved]
        default_weather = ["temperature", "irradiance", "wind_speed"]

        if self.explicit_target_cols is not None:
            target_cols = self.explicit_target_cols
        elif self.task_mode == "net_injection":
            if "p_vpp_mw" in df_raw.columns:
                target_cols = ["p_vpp_mw"]
            else:
                raise ValueError("task_mode='net_injection' requires target_cols or a 'p_vpp_mw' column.")
        else:
            target_cols = [c for c in ["load_mw", "pv_mw", "wind_mw"] if c in df_raw.columns]
            if not target_cols:
                target_cols = data_cols[:3]

        if self.explicit_known_future_covariate_cols is not None:
            known_future_covariate_cols = self.explicit_known_future_covariate_cols
        elif self.legacy_covariate_cols is not None:
            known_future_covariate_cols = self.legacy_covariate_cols
        else:
            known_future_covariate_cols = [c for c in default_weather if c in df_raw.columns and c not in target_cols]

        history_state_cols = [c for c in self.explicit_history_state_cols if c]
        aux_target_cols = [c for c in self.explicit_aux_target_cols if c]

        missing = [
            c
            for c in target_cols + known_future_covariate_cols + history_state_cols + aux_target_cols
            if c not in df_raw.columns
        ]
        if missing:
            raise ValueError(f"Missing configured columns in dataset: {missing}")

        feature_cols = list(dict.fromkeys(target_cols + known_future_covariate_cols + history_state_cols))
        return target_cols, known_future_covariate_cols, history_state_cols, aux_target_cols, feature_cols

    def _split_one_group(self, df_group):
        n = len(df_group)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_val = n - num_train - num_test

        border1s = [0, max(0, num_train - self.seq_len), max(0, n - num_test - self.seq_len)]
        border2s = [num_train, num_train + num_val, n]
        return border1s[self.set_type], border2s[self.set_type], border1s[0], border2s[0]

    def __read_data__(self):
        full_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(full_path)
        if self.time_col not in df_raw.columns:
            raise ValueError(f"Missing time column '{self.time_col}' in {full_path}")

        df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col])
        sort_cols = [self.id_col, self.time_col] if self.id_col and self.id_col in df_raw.columns else [self.time_col]
        df_raw = df_raw.sort_values(sort_cols).reset_index(drop=True)

        (
            self.target_cols,
            self.known_future_covariate_cols,
            self.history_state_cols,
            self.aux_target_cols,
            self.feature_cols,
        ) = self._resolve_columns(df_raw)

        self.target_num = len(self.target_cols)
        self.known_future_num = len(self.known_future_covariate_cols)
        self.history_state_num = len(self.history_state_cols)
        self.aux_target_num = len(self.aux_target_cols)

        manifest_mode = self.split_strategy == "portfolio_manifest" and self.split_col and self.split_col in df_raw.columns

        train_feature_frames = []
        train_aux_frames = []
        prepared_groups = []

        if manifest_mode:
            split_value = ["train", "val", "test"][self.set_type]
            train_frame = df_raw.loc[df_raw[self.split_col] == "train"].copy()
            if train_frame.empty:
                raise ValueError(
                    f"split_strategy='portfolio_manifest' requires at least one train row in split column '{self.split_col}'."
                )
            train_feature_frames.append(train_frame[self.feature_cols])
            if self.aux_target_cols:
                train_aux_frames.append(train_frame[self.aux_target_cols])

            active_frame = df_raw.loc[df_raw[self.split_col] == split_value].copy()
            if self.id_col and self.id_col in active_frame.columns:
                grouped = [(str(group_id), df_group.copy()) for group_id, df_group in active_frame.groupby(self.id_col, sort=False)]
            else:
                grouped = [("__global__", active_frame.copy())]

            for group_id, df_group in grouped:
                df_group = df_group.sort_values(self.time_col).reset_index(drop=True)
                prepared_groups.append((group_id, df_group, 0, len(df_group)))
        else:
            if self.id_col and self.id_col in df_raw.columns:
                grouped = [(str(group_id), df_group.copy()) for group_id, df_group in df_raw.groupby(self.id_col, sort=False)]
            else:
                grouped = [("__global__", df_raw.copy())]

            for group_id, df_group in grouped:
                df_group = df_group.sort_values(self.time_col).reset_index(drop=True)
                border1, border2, train_border1, train_border2 = self._split_one_group(df_group)
                train_feature_frames.append(df_group.loc[train_border1:train_border2 - 1, self.feature_cols])
                if self.aux_target_cols:
                    train_aux_frames.append(df_group.loc[train_border1:train_border2 - 1, self.aux_target_cols])
                prepared_groups.append((group_id, df_group, border1, border2))

        train_feature_frame = pd.concat(train_feature_frames, axis=0, ignore_index=True)
        self._train_feature_frame = train_feature_frame.copy()
        if self.scale:
            self.feature_scaler.fit(train_feature_frame.values)
        else:
            self.feature_scaler = None

        if self.aux_target_cols:
            train_aux_frame = pd.concat(train_aux_frames, axis=0, ignore_index=True)
            self._train_aux_frame = train_aux_frame.copy()
            self.aux_scaler.fit(train_aux_frame.values)
        else:
            self._train_aux_frame = None
            self.aux_scaler = None

        for group_id, df_group, border1, border2 in prepared_groups:
            feature_frame = df_group[self.feature_cols]
            feature_data = self.feature_scaler.transform(feature_frame.values) if self.scale else feature_frame.values

            if self.aux_target_cols:
                aux_frame = df_group[self.aux_target_cols]
                aux_data = self.aux_scaler.transform(aux_frame.values) if self.scale else aux_frame.values
            else:
                aux_data = np.zeros((len(df_group), 0), dtype=np.float32)

            df_stamp = df_group[[self.time_col]].iloc[border1:border2].copy()
            data_stamp = time_features(df_stamp[self.time_col].values, freq="t")

            feature_slice = feature_data[border1:border2]
            aux_slice = aux_data[border1:border2]
            if len(feature_slice) == 0:
                continue

            self.group_ids.append(group_id)
            region_value = df_group[self.region_col].iloc[0] if self.region_col and self.region_col in df_group.columns else None
            self.group_region_ids.append(region_value)
            self.group_x_tensors.append(torch.tensor(feature_slice, dtype=torch.float32))
            self.group_y_tensors.append(torch.tensor(feature_slice, dtype=torch.float32))
            self.group_aux_tensors.append(torch.tensor(aux_slice, dtype=torch.float32))
            self.group_stamp_tensors.append(torch.tensor(data_stamp, dtype=torch.float32))
            self.group_time_indices.append(pd.to_datetime(df_stamp[self.time_col].values))

        for group_idx, group_tensor in enumerate(self.group_x_tensors):
            available = len(group_tensor) - self.seq_len - self.pred_len + 1
            if available <= 0:
                continue
            for start_idx in range(available):
                self.sample_index.append((group_idx, start_idx))

        if self._train_feature_frame is not None:
            for col in self._train_feature_frame.columns:
                self.col_ranges[col] = self._train_feature_frame[col].max() - self._train_feature_frame[col].min()

    def __getitem__(self, index):
        group_idx, s_begin = self.sample_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.group_x_tensors[group_idx][s_begin:s_end]
        seq_y = self.group_y_tensors[group_idx][r_begin:r_end]
        seq_x_mark = self.group_stamp_tensors[group_idx][s_begin:s_end]
        seq_y_mark = self.group_stamp_tensors[group_idx][r_begin:r_end]

        if self.set_type == 0 and self.noise_level > 0 and self.known_future_num > 0:
            seq_x = seq_x.clone()
            cov_start = self.target_num
            cov_end = cov_start + self.known_future_num
            seq_x[:, cov_start:cov_end] += torch.randn_like(seq_x[:, cov_start:cov_end]) * self.noise_level

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_index)

    def inverse_transform(self, data):
        if not self.scale or self.feature_scaler is None:
            return data

        if torch.is_tensor(data):
            data = data.cpu().numpy()

        original_shape = data.shape
        if len(original_shape) == 3:
            data = data.reshape(-1, original_shape[-1])

        n_expected = self.feature_scaler.n_features_in_
        n_input = data.shape[1]
        if n_input == n_expected:
            data_rescaled = self.feature_scaler.inverse_transform(data)
        elif n_input < n_expected:
            dummy = np.zeros((data.shape[0], n_expected))
            dummy[:, :n_input] = data
            data_rescaled = self.feature_scaler.inverse_transform(dummy)[:, :n_input]
        else:
            raise ValueError(f"Input features ({n_input}) > scaler features ({n_expected})")

        if len(original_shape) == 3:
            data_rescaled = data_rescaled.reshape(original_shape)
        return data_rescaled

    def inverse_transform_aux(self, data):
        if not self.scale or self.aux_scaler is None or self.aux_target_num == 0:
            return data

        if torch.is_tensor(data):
            data = data.cpu().numpy()

        original_shape = data.shape
        if len(original_shape) == 3:
            data = data.reshape(-1, original_shape[-1])

        data_rescaled = self.aux_scaler.inverse_transform(data)
        if len(original_shape) == 3:
            data_rescaled = data_rescaled.reshape(original_shape)
        return data_rescaled

    def get_prediction_metadata(self, index):
        group_idx, s_begin = self.sample_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        times = self.group_time_indices[group_idx][r_end - self.pred_len : r_end]
        return {
            "portfolio_id": self.group_ids[group_idx],
            "region_id": self.group_region_ids[group_idx],
            "forecast_timestamps": pd.to_datetime(times),
        }

    def get_schema(self):
        weather_start = self.target_num
        weather_end = weather_start + self.known_future_num
        state_start = weather_end
        state_end = state_start + self.history_state_num
        return {
            "task_mode": self.task_mode,
            "time_col": self.time_col,
            "id_col": self.id_col,
            "region_col": self.region_col,
            "split_col": self.split_col,
            "split_strategy": self.split_strategy,
            "target_cols": self.target_cols.copy(),
            "known_future_covariate_cols": self.known_future_covariate_cols.copy(),
            "history_state_cols": self.history_state_cols.copy(),
            "aux_target_cols": self.aux_target_cols.copy(),
            "feature_cols": self.feature_cols.copy(),
            "feature_slices": {
                "target": [0, self.target_num],
                "known_future_covariates": [weather_start, weather_end],
                "history_state": [state_start, state_end],
            },
        }


class PhysFormerDataset(VPPDataset):
    """
    Thesis-only PhysFormer dataset for net injection + auxiliary supervision.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.task_mode != "net_injection":
            raise ValueError("PhysFormerDataset now supports only task_mode='net_injection'.")
        if self.target_num != 1:
            raise ValueError("PhysFormerDataset requires exactly one main target column.")
        if self.known_future_num < 1:
            raise ValueError("PhysFormerDataset requires at least one known future covariate.")
        if self.history_state_num < 1:
            raise ValueError("PhysFormerDataset requires history_state_cols for battery state input.")
        if self.aux_target_num < 1:
            raise ValueError("PhysFormerDataset requires auxiliary component/battery targets.")

    def get_scaler_params(self):
        if not self.scale or self.feature_scaler is None:
            return {
                "target_mean": None,
                "target_std": None,
                "weather_mean": None,
                "weather_std": None,
                "state_mean": None,
                "state_std": None,
                "aux_mean": None,
                "aux_std": None,
            }

        weather_start = self.target_num
        weather_end = weather_start + self.known_future_num
        state_start = weather_end
        state_end = state_start + self.history_state_num
        return {
            "target_mean": self.feature_scaler.mean_[: self.target_num].copy(),
            "target_std": self.feature_scaler.scale_[: self.target_num].copy(),
            "weather_mean": self.feature_scaler.mean_[weather_start:weather_end].copy(),
            "weather_std": self.feature_scaler.scale_[weather_start:weather_end].copy(),
            "state_mean": self.feature_scaler.mean_[state_start:state_end].copy(),
            "state_std": self.feature_scaler.scale_[state_start:state_end].copy(),
            "aux_mean": self.aux_scaler.mean_.copy() if self.aux_scaler is not None else None,
            "aux_std": self.aux_scaler.scale_.copy() if self.aux_scaler is not None else None,
        }

    def get_training_statistics(self):
        stats = self.get_scaler_params()
        if self._train_feature_frame is not None and self.target_cols:
            target_diff = self._train_feature_frame[self.target_cols].diff().abs().dropna()
            if not target_diff.empty:
                stats["net_ramp_limit"] = float(target_diff.quantile(0.999).iloc[0] * 1.5)
            else:
                stats["net_ramp_limit"] = 0.0
        else:
            stats["net_ramp_limit"] = 0.0

        if self._train_aux_frame is not None and "p_battery_mw" in self._train_aux_frame.columns:
            battery_diff = self._train_aux_frame[["p_battery_mw"]].diff().abs().dropna()
            stats["battery_ramp_limit"] = float(battery_diff.quantile(0.999).iloc[0] * 1.5) if not battery_diff.empty else 0.0
        else:
            stats["battery_ramp_limit"] = 0.0

        return stats

    def __getitem__(self, index):
        group_idx, s_begin = self.sample_index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_raw = self.group_x_tensors[group_idx][s_begin:s_end]
        seq_y_raw = self.group_y_tensors[group_idx][r_begin:r_end]
        seq_aux_raw = self.group_aux_tensors[group_idx][r_begin:r_end]

        weather_start = self.target_num
        weather_end = weather_start + self.known_future_num
        state_start = weather_end
        state_end = state_start + self.history_state_num

        x_net_hist = seq_raw[:, : self.target_num]
        x_weather_hist = seq_raw[:, weather_start:weather_end]
        x_battery_hist = seq_raw[:, state_start:state_end]
        x_weather_future = seq_y_raw[-self.pred_len :, weather_start:weather_end]
        y_target = seq_y_raw[-self.pred_len :, : self.target_num]
        y_aux = seq_aux_raw[-self.pred_len :, :]

        x_mark_enc = self.group_stamp_tensors[group_idx][s_begin:s_end]
        y_mark = self.group_stamp_tensors[group_idx][r_end - self.pred_len : r_end]

        if self.set_type == 0 and self.noise_level > 0:
            x_weather_hist = x_weather_hist.clone() + torch.randn_like(x_weather_hist) * self.noise_level
            x_weather_future = x_weather_future.clone() + torch.randn_like(x_weather_future) * self.noise_level

        return x_net_hist, x_weather_hist, x_battery_hist, x_weather_future, y_target, y_aux, x_mark_enc, y_mark


def data_provider(args, flag):
    Data = PhysFormerDataset if args.model == "PhysFormer" else VPPDataset

    shuffle_flag = flag == "train"
    drop_last = flag == "train"
    batch_size = args.batch_size if flag != "test" else 1
    num_workers = getattr(args, "num_workers", 0)
    pin_memory = bool(getattr(args, "pin_memory", False)) and bool(getattr(args, "use_gpu", False))
    persistent_workers = bool(getattr(args, "persistent_workers", False)) and num_workers > 0
    prefetch_factor = getattr(args, "prefetch_factor", 2)

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        noise_level=0.03 if flag == "train" else 0.0,
        time_col=getattr(args, "time_col", "date"),
        id_col=getattr(args, "id_col", None),
        region_col=getattr(args, "region_col", None),
        split_col=getattr(args, "split_col", None),
        split_strategy=getattr(args, "split_strategy", "time_series"),
        target_cols=getattr(args, "target_cols", None),
        covariate_cols=getattr(args, "covariate_cols", None),
        known_future_covariate_cols=getattr(args, "known_future_covariate_cols", None),
        history_state_cols=getattr(args, "history_state_cols", None),
        aux_target_cols=getattr(args, "aux_target_cols", None),
        task_mode=getattr(args, "task_mode", "component_multitask"),
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    data_loader = DataLoader(
        data_set,
        **loader_kwargs,
    )
    return data_set, data_loader
