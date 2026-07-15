from physformer.config import config_to_args, finalize_args, load_config
from physformer.data import data_provider

import argparse
import numpy as np


cfg = load_config("configs/baselines/itransformer_net_injection.yaml")
args = config_to_args(cfg)
cli = argparse.Namespace(
    epochs=None,
    lr=None,
    batch_size=256,
    num_workers=0,
    patience=None,
    gpu=None,
    seed=None,
    run_name="tmp",
    run_dir=None,
    resume=False,
    debug_nan=False,
    save_gate_details=False,
    init_from_run=None,
    disable_fused_rnn_backends=False,
)
args, cfg = finalize_args(args, cfg, cli)
args.use_gpu = False

data, loader = data_provider(args, "test")
target_batches = []
weather_batches = []
hist_last_batches = []
mark_batches = []
count = 0

for bx, by, bxm, bym in loader:
    target_batches.append(by[:, -args.pred_len :, :1].numpy())
    weather_batches.append(by[:, -args.pred_len :, 1:4].numpy())
    hist_last_batches.append(bx[:, -1:, :1].numpy().repeat(args.pred_len, axis=1))
    mark_batches.append(bym[:, -args.pred_len :, :].numpy())
    count += len(bx)
    if count >= 4096:
        break

target = np.concatenate(target_batches)
weather = np.concatenate(weather_batches)
hist_last = np.concatenate(hist_last_batches)
marks = np.concatenate(mark_batches)

target_raw = data.inverse_transform(target)
hist_raw = data.inverse_transform(hist_last)

print("feature_cols", data.feature_cols)
print("target shape", target.shape)
print(
    "target raw mean/std/min/max",
    float(target_raw.mean()),
    float(target_raw.std()),
    float(target_raw.min()),
    float(target_raw.max()),
)
print("norm target std", float(target.std()))
print("persistence subset mae", float(np.mean(np.abs(hist_raw - target_raw))))
print("persistence corr", float(np.corrcoef(hist_raw.reshape(-1), target_raw.reshape(-1))[0, 1]))

flat_t = target.reshape(-1)
for i, name in enumerate(args.known_future_covariate_cols):
    corr = float(np.corrcoef(weather[..., i].reshape(-1), flat_t)[0, 1])
    print("norm corr weather", name, corr)
for i in range(marks.shape[-1]):
    corr = float(np.corrcoef(marks[..., i].reshape(-1), flat_t)[0, 1])
    print("norm corr mark", i, corr)
