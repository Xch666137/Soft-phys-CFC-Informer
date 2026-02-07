import torch
import torch.optim as optim
import numpy as np
import os
import time
import warnings
import logging

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_loader.data_factory import PhysFormerDataset
from models.src.models.PhysFormer.model import PhysFormer
from models.src.utils.losses import PhysAwareVPPLoss
from models.src.utils.metrics import metric

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

class Exp_PhysFormer:
    def __init__(self, args):
        self.args = args
        self._init_logger()
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        print(f"PhysFormer Model Structure:\n{self.model}")

    def _init_logger(self):
        log_dir = os.path.join(self.args.checkpoints, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 区分日志文件名，避免覆盖
        log_file = os.path.join(log_dir, f'train_log_{self.args.checkpoint_name}.txt')

        self.logger = logging.getLogger("VPP_CFC_Informer")
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info(f"Log file created at: {log_file}")

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device

    def _build_model(self):
        # --- 核心修改：构建 CFCInformer 模型 ---
        model = PhysFormer(
            enc_in=self.args.enc_in,
            dec_in=self.args.dec_in,
            c_out=self.args.c_out,
            seq_len=self.args.seq_len,
            label_len=self.args.label_len,
            pred_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_layers=self.args.d_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            attn=self.args.attn,
            embed=self.args.embed,
            freq=self.args.freq,
            activation=self.args.activation,
            output_attention=self.args.output_attention,
            distil=self.args.distil,
            mix=self.args.mix,
            device=self.device,
            use_rope=self.args.use_rope,
            rope_base=self.args.rope_base,
            stride=self.args.stride
        )

        return model

    def _get_data(self, flag):
        # 数据加载逻辑与原版完全一致
        data_set = PhysFormerDataset(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            scale=True,
            target=None,
            noise_level=0.03
        )

        shuffle_flag = True if flag == 'train' else False
        drop_last = True
        batch_size = self.args.batch_size
        use_persistent = self.args.num_workers > 0

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
            persistent_workers=use_persistent,
            pin_memory=True if self.args.use_gpu else False
        )
        return data_set, data_loader

    def _select_criterion(self):
        # -----------------------------------------------------------
        # [动态统计] 从训练集获取真实的 Means, Stds 和 Ramp Limits
        # -----------------------------------------------------------

        # 1. 获取训练集 (复用已有的逻辑，避免重复加载)
        if not hasattr(self, 'train_data_scaler'):
            train_data, _ = self._get_data(flag='train')
            self.train_data_scaler = train_data.scaler
            # 保存训练数据的引用以便计算 Ramp
            self.train_data_obj = train_data

        # 2. 提取统计量 (Means, Stds)
        if self.train_data_scaler is not None:
            # scaler.mean_ 和 scale_ 对应所有特征，我们只取前3个 (Load, PV, Wind)
            real_means = self.train_data_scaler.mean_[:3]
            real_stds = self.train_data_scaler.scale_[:3]
            self.logger.info(f">>> [Auto-Stat] Loaded Real Means: {real_means}")
            self.logger.info(f">>> [Auto-Stat] Loaded Real Stds : {real_stds}")
        else:
            real_means, real_stds = None, None

        # 3. 计算真实的爬坡极限 (Ramp Limits)
        # 逻辑：取训练集一阶差分的 99.9% 分位数
        try:
            # 获取原始归一化数据 [Seq_Len, Features]
            # 注意：这里假设 Dataset 内部有 data_x 属性 (标准 Informer 代码结构都有)
            raw_data = self.train_data_obj.data_x

            # 反归一化 (还原为 MW)
            if self.train_data_scaler:
                raw_data = self.train_data_scaler.inverse_transform(raw_data)

            # 只取前3列 (Load, PV, Wind)
            target_data = raw_data[:, :3]

            # 计算差分 |t - (t-1)|
            diff = np.abs(target_data[1:] - target_data[:-1])

            # 计算 99.5% 分位数作为极限 (留0.5%给极端异常值)
            real_ramp_limits = np.percentile(diff, 99.9, axis=0)

            # 稍微给一点点裕度 (1.0~1.5倍)，防止训练太敏感
            # 逻辑：我们只拦截“完全不可能”的物理错误（如数据跳变、传感器故障），
            # 而不是去拦截“罕见但真实”的极端天气。
            real_ramp_limits = real_ramp_limits * 1.5

            self.logger.info(f">>> [Auto-Stat] Calculated Ramp Limits: {real_ramp_limits}")

        except Exception as e:
            self.logger.warning(f"Failed to calc ramp limits: {e}, using defaults.")
            real_ramp_limits = None

        # 将统计量保存到 self，供 test 阶段复用
        self.train_means = real_means
        self.train_stds = real_stds
        self.train_ramp_limits = real_ramp_limits

        # 4. 将真实统计量传入 Loss
        criterion = PhysAwareVPPLoss(
            device=self.device,
            means=real_means,
            stds=real_stds,
            ramp_limits=real_ramp_limits
        )
        return criterion

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-4

        params = list(self.model.parameters())

        if isinstance(self.criterion, torch.nn.Module):
            loss_params = list(self.criterion.parameters())
            params += loss_params
            self.logger.info(f">>> Optimizer: Added {len(loss_params)} params from Criterion (Auto-Weighting).")

        model_optim = optim.AdamW(
            params,
            lr=self.args.learning_rate,
            weight_decay=wd
        )
        return model_optim

    def _process_one_batch(self, batch_data, criterion=None, phase='train'):
        """
        统一处理一个 Batch 的数据准备、模型前向传播和 Loss 计算
        Args:
            batch_data: DataLoader 吐出的 tuple (batch_stat, batch_phys, batch_y, ...)
            criterion: 损失函数 (仅 train/val 需要)
            phase: 'train', 'val', 'test'
        Returns:
            outputs: 模型输出
            batch_y_true: 真实标签 (仅预测部分)
            loss: 总损失 (如果提供了 criterion)
            loss_dict: 物理损失字典
        """
        # 1. 解包数据并转移到 GPU
        batch_stat, batch_phys, batch_y, batch_x_mark, batch_y_mark = batch_data

        batch_stat = batch_stat.float().to(self.device)
        batch_phys = batch_phys.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # 2. 构造 Decoder Input (核心修改：引入未来天气驱动)
        # 假设 batch_y 的列顺序: [0:Load, 1:PV, 2:Wind, 3:Temp, 4:Irr, 5:Speed]

        # Part A: Label (历史已知部分) -> 包含 3个功率 + 3个天气
        # Shape: [Batch, Label_Len, 6]
        label_part = batch_y[:, :self.args.label_len, :]

        # Part B: Prediction (未来部分)
        # B.1: 功率 (待预测) -> 填 0
        future_power_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :self.args.c_out])

        # B.2: 天气 (未来已知) -> 从 batch_y 截取
        # 注意：这里假设 Dataset 确实返回了 6 列数据 (c_out=3, 总列数=6)
        # 如果你的 args.c_out 是 3，那么 weather 就是从 index 3 开始
        future_weather_known = batch_y[:, -self.args.pred_len:, self.args.c_out:]

        # B.3: 拼接 -> [0, 0, 0, T, I, S]
        pred_part = torch.cat([future_power_zeros, future_weather_known], dim=-1)

        # Part C: 最终拼接
        dec_inp = torch.cat([label_part, pred_part], dim=1).float().to(self.device)

        # 3. 前向传播 (混合精度)
        # 验证和测试时如果不需反向传播，理论上也可以不用 autocast，但为了保持一致性建议加上
        use_amp = self.args.use_amp and (phase != 'test_speed')

        # 针对 train/val/test 的上下文管理
        context = torch.cuda.amp.autocast(enabled=use_amp) if self.args.use_gpu else torch.no_grad()

        # 如果是 eval 模式，通常不需要梯度
        if phase in ['val', 'test']:
            # 这里的上下文稍微复杂点，直接在外部控制 torch.no_grad() 更方便
            # 但为了函数独立性，我们在内部只处理 autocast
            pass

        # 4. 准备 Ground Truth
        # 只取前 c_out 列 (功率) 进行 Loss 计算
        outputs = None
        batch_y_true = batch_y[:, -self.args.pred_len:, :self.args.c_out]

        # 5. 计算 Loss
        total_loss = None
        loss_dict = {}

        with torch.cuda.amp.autocast(enabled=self.args.use_amp):
            if self.args.output_attention:
                outputs = self.model(batch_stat, batch_phys, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_stat, batch_phys, batch_x_mark, dec_inp, batch_y_mark)

            # 将 criterion 放入 autocast 内部，确保类型对齐
            if criterion is not None:
                # 计算主 Loss (MSE + 物理约束)
                loss_main, loss_dict = criterion(outputs, batch_y_true)
                total_loss = loss_main

        return outputs, batch_y_true, total_loss, loss_dict

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, self.args.checkpoint_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # 1. 初始化 Loss (获取 log_vars 参数)
        self.criterion = self._select_criterion()

        # 2. 再初始化 Optimizer (传入 Model + Loss 的参数)
        model_optim = self._select_optimizer()

        # 3. 早停机制 (监控 物理MAE)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

        # 4. 学习率调度器: 余弦退火
        scheduler = CosineAnnealingLR(
            model_optim,
            T_max=self.args.train_epochs,
            eta_min=1e-6                    # 最小保底 LR
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            self.criterion.train()

            train_loss_log = []
            epoch_time = time.time()

            with tqdm(total=len(train_loader),
                      desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                      mininterval=0.3,
                      leave=False,
                      ncols=100) as pbar:

                for i, batch_data in enumerate(train_loader):
                    model_optim.zero_grad()

                    # Train 阶段我们还需要加上惯性正则化 Loss，所以我们在外层加
                    outputs, batch_y_true, loss_main, loss_dict = self._process_one_batch(
                        batch_data, self.criterion, phase='train'
                    )

                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                        # 注意使用 self.args.w_inertia
                        loss_inertia = calculate_inertia_loss(self.model, lambda_inertia=self.args.w_inertia)
                        total_loss = loss_main + loss_inertia

                    # Backward
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()

                    train_loss_log.append(total_loss.item())
                    pbar.update(1)

            # --- C. 验证与早停 ---
            # 计算并输出单轮训练耗时
            epoch_cost_time = time.time() - epoch_time

            # 这里的 vali_mae 是反归一化后的真实物理误差 (MW)
            vali_loss, vali_mae = self.vali(val_data, val_loader, self.criterion)

            # 调度器步进 (WarmRestarts 是按 epoch 更新的)
            scheduler.step()

            # 获取当前所有权重 (从最后一个 batch 的 loss_dict 中取)
            # 使用 .get() 给一个默认值，防止第一轮没跑完报错
            w_base = loss_dict.get('w_base', 0)
            w_net = loss_dict.get('w_net', 0)
            w_deriv = loss_dict.get('w_deriv', 0)
            w_energy = loss_dict.get('w_energy', 0)
            w_dir = loss_dict.get('w_dir', 0)
            w_cons = loss_dict.get('w_cons', 0)

            self.logger.info(
                f"Epoch {epoch + 1} | Train Loss: {np.mean(train_loss_log):.5f} | Vali MAE: {vali_mae:.5f}")
            # 打印全量权重看板
            self.logger.info(f"  >> [Weights 1] Base: {w_base:.2f} | Net: {w_net:.2f} | Deriv: {w_deriv:.2f}")
            self.logger.info(f"  >> [Weights 2] Energy: {w_energy:.2f} | Dir: {w_dir:.2f} | Cons: {w_cons:.2f}")

            # 早停判断
            early_stopping(vali_mae, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping based on Physical MAE (Constraints Locked)")
                break

        # Load Best Model
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        self.criterion.eval()

        total_loss = []

        # 用于记录各项物理违规的平均值
        total_metrics = {'mae': [], 'net': [], 'deriv': [], 'energy': [], 'cons': []}

        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                outputs, batch_y_true, loss, loss_dict = self._process_one_batch(
                    batch_data, criterion, phase='val'
                )

                total_loss.append(loss.item())

                # 累积各项指标
                for k, v in loss_dict.items():
                    if k in total_metrics:
                        total_metrics[k].append(v)

        avg_loss = np.average(total_loss)
        avg_mae = np.average(total_metrics['mae'])

        self.model.train()
        self.criterion.train()

        # 返回 avg_loss 用于日志，avg_mae 用于早停 (EarlyStopping)
        return avg_loss, avg_mae

    def test(self, setting, load=True):
        test_data, test_loader = self._get_data(flag='test')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        preds = []
        trues = []

        # 使用 PhysAwareVPPLoss，并传入训练时的统计量
        # 如果 self.train_means 不存在 (比如直接运行 test 没经过 train)，则需要重新计算或加载
        if not hasattr(self, 'train_means') or self.train_means is None:
            self.logger.warning("Warning: Training stats not found. Re-calculating from train set...")
            _ = self._select_criterion()  # 重新触发一次统计计算

        criterion = PhysAwareVPPLoss(
            device=self.device,
            means=self.train_means,
            stds=self.train_stds,
            ramp_limits=self.train_ramp_limits
        )

        # 记录详细物理合规性
        phys_metrics = {'base': [], 'net': [], 'deriv': [], 'energy': [], 'dir': [], 'cons': []}

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                outputs, batch_y_true, _, loss_dict = self._process_one_batch(
                    batch_data, criterion, phase='test'
                )

                # 记录指标
                for k in phys_metrics.keys():
                    if k in loss_dict:
                        phys_metrics[k].append(loss_dict[k])

                # [Batch, Pred_Len, 3]
                pred = outputs.detach().cpu().numpy()
                true = batch_y_true.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

        # 拼接所有 Batch
        preds = np.concatenate(preds, axis=0)  # [N, P, 3]
        trues = np.concatenate(trues, axis=0)  # [N, P, 3]

        # --- 反归一化逻辑 ---
        if test_data.scale and test_data.scaler is not None:
            # 展平 [N*P, 3]
            shape_orig = preds.shape
            preds_2d = preds.reshape(-1, shape_orig[-1])
            trues_2d = trues.reshape(-1, shape_orig[-1])

            # 使用 Dataset 中自定义的 smart inverse_transform (支持自动补全列)
            # 它会自动把 3列 补全成 6列，反归一化后再切回 3列
            preds_rescaled = test_data.inverse_transform(preds_2d)
            trues_rescaled = test_data.inverse_transform(trues_2d)

            # 恢复形状
            preds = preds_rescaled.reshape(shape_orig)
            trues = trues_rescaled.reshape(shape_orig)

        # 调用 metrics.py 计算 7 大指标
        metrics_result = metric(preds, trues)
        # metric函数返回：mae, mse, rmse, mape, mspe, bvr, rvr

        # 打印最终物理合规性报告
        print("\n" + "=" * 50)
        print("  VPP Physics Compliance Report (Test Set)")
        print("=" * 50)
        print(f"  Net Load Error (MW) : {np.mean(phys_metrics['net']):.6f}")
        print(f"  Energy Deviation    : {np.mean(phys_metrics['energy']):.6f}")
        print(f"  Shape/Deriv Error   : {np.mean(phys_metrics['deriv']):.6f}")
        print(f"  Direction Failures  : {np.mean(phys_metrics['dir']):.6f}")
        print(f"  Hard Constraints    : {np.mean(phys_metrics['cons']):.6f}")
        print("=" * 50 + "\n")

        # 结果保存
        path = os.path.join(self.args.checkpoints, self.args.checkpoint_name)
        if not os.path.exists(path):
            os.makedirs(path)

        folder_path = path + '/'

        # 保存的就是真实的 MW 值了
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array(metrics_result))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return preds, trues


def calculate_inertia_loss(model, lambda_inertia=0.01):
    """
    计算 CFC 层的参数正则化损失，强迫模型学习平滑的动力学特性
    """
    # nn.Module 没有 .device 属性，通过第一个参数来判断设备
    device = next(model.parameters()).device
    inertia_loss = torch.tensor(0.0, device=device)

    # 获取 CFC 模块 (假设你的模型里叫 physics_adapter)
    cfc_block = model.physics_adapter

    # 锁定控制时间演化的关键层
    # 这些层的权重决定了 ODE 的 "Time Scale"
    target_layers = [
        cfc_block.x_time_a, cfc_block.h_time_a,
        cfc_block.x_time_b, cfc_block.h_time_b
    ]

    for layer in target_layers:
        # L2 Regularization on Time-Control Weights
        inertia_loss += torch.norm(layer.weight, p=2)

    return lambda_inertia * inertia_loss

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.logger = logger  # 传入 logger

    def __call__(self, val_loss, model, path):
        # 这里的 val_loss 实际上是 NRMSE，越小越好
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.logger:
                self.logger.info(
                    f'EarlyStopping counter: {self.counter} out of {self.patience} (Best: {-self.best_score:.6f})')
            elif self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.logger:
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        elif self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss