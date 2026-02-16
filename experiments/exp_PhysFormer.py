import torch
import torch.optim as optim
import numpy as np
import os
import warnings
import logging
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_loader.data_factory import PhysFormerDataset
from models.src.models import PhysFormer
from models.src.utils.losses import PhysAwareBaseLoss, PhysLoss
from models.src.utils.metrics import metric

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

class Exp_PhysFormer:
    def __init__(self, args):
        self.args = args
        self._init_logger()
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.gate_history = {
            'epoch': [],
            'load': [],
            'pv': [],
            'wind': [],
            'info_gain': []
        }

        # 初始化 TensorBoard Writer
        # 日志保存在 checkpoints/你的实验名/tensorboard 目录下
        tb_dir = os.path.join(self.args.checkpoints, self.args.checkpoint_name, 'tensorboard')
        self.writer = SummaryWriter(log_dir=tb_dir)
        print(f">>> TensorBoard logging to: {tb_dir}")

        print(f"PhysFormer Model Structure:\n{self.model}")

    # 记录显式物理参数
    def _log_physical_params(self, global_step):
        # 获取物理层句柄
        pl = self.model.phys_layer

        # 注意：必须应用与 forward 中相同的激活函数 (Softplus/Sigmoid) 才能看到真实物理值
        with torch.no_grad():
            # PV 参数
            pv_eff = torch.nn.functional.softplus(pl.pv_efficiency)
            pv_temp_coef = torch.nn.functional.softplus(pl.pv_temp_coef) * 0.01

            # Wind 参数
            wind_scale = torch.nn.functional.softplus(pl.wind_scale)
            # Load 参数
            load_base = pl.load_base
            load_sens = torch.nn.functional.softplus(pl.load_temp_sens)

            # 写入 TensorBoard
            self.writer.add_scalar('Physics_Params/PV_Efficiency', pv_eff, global_step)
            self.writer.add_scalar('Physics_Params/PV_Temp_Coef', pv_temp_coef, global_step)
            self.writer.add_scalar('Physics_Params/Wind_Scale', wind_scale, global_step)
            self.writer.add_scalar('Physics_Params/Wind_Cut_In', pl.wind_cut_in, global_step)
            self.writer.add_scalar('Physics_Params/Load_Base', load_base, global_step)
            self.writer.add_scalar('Physics_Params/Load_Temp_Sens', load_sens, global_step)

    # [新增方法] 记录梯度范数 (检查物理层是否在学习)
    def _log_gradient_norms(self, global_step):
        total_norm = 0.0
        phys_norm = 0.0
        enc_norm = 0.0

        for name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                if 'phys_layer' in name:
                    phys_norm += param_norm ** 2
                elif 'encoder' in name:
                    enc_norm += param_norm ** 2

        total_norm = total_norm ** 0.5
        phys_norm = phys_norm ** 0.5
        enc_norm = enc_norm ** 0.5

        self.writer.add_scalar('Gradients/Total_Norm', total_norm, global_step)
        self.writer.add_scalar('Gradients/Phys_Layer_Norm', phys_norm, global_step)
        self.writer.add_scalar('Gradients/Encoder_Norm', enc_norm, global_step)

    def _plot_prediction(self, preds, trues, dataset):
        """
        绘制预测对比图 (取 Batch 中的第一个样本)
        """
        # 1. 转移到 CPU 并转为 Numpy
        preds = preds.detach().cpu().numpy()  # [B, Pred, 3]
        trues = trues.detach().cpu().numpy()  # [B, Pred, 3]

        # 2. 反归一化 (还原为真实物理量 MW)
        # 利用 Dataset 中已经写好的 inverse_transform，它会自动处理 shape
        if dataset.scale:
            preds = dataset.inverse_transform(preds)
            trues = dataset.inverse_transform(trues)

        # 3. 取第一个样本 (Sample 0)
        # shape 变成 [Pred, 3]
        pred_sample = preds[0]
        true_sample = trues[0]

        # 4. 绘图 (3行1列)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        titles = ['Load (MW)', 'PV Power (MW)', 'Wind Power (MW)']
        colors = ['blue', 'orange', 'green']

        for i in range(3):
            ax = axs[i]
            # 画真实值 (黑色实线)
            ax.plot(true_sample[:, i], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
            # 画预测值 (彩色虚线)
            ax.plot(pred_sample[:, i], label='Prediction', color=colors[i], linestyle='--', linewidth=2.0)

            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.set_ylabel('Power')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.xlabel('Time Steps (Future)')
        plt.tight_layout()

        return fig

    def _init_logger(self):
        log_dir = os.path.join(self.args.checkpoints, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 区分日志文件名，避免覆盖
        log_file = os.path.join(log_dir, f'train_log_{self.args.checkpoint_name}.txt')

        self.logger = logging.getLogger("PhysFormer")
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

    def _get_data(self, flag):
        # 1. 动态参数设置
        if flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size
            noise_level = 0.03
        else:
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size
            noise_level = 0.0

        # 2. 实例化 Dataset
        # size=[seq_len, 0, pred_len]
        # PhysFormer 是 Encoder-Only 模型，不需要 label_len (Start Token)
        # 将其设为 0，Dataset 就会直接从 seq_len 结束的位置开始取 pred_len 长度的标签
        data_set = PhysFormerDataset(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, 0, self.args.pred_len],  # label_len 硬编码为 0
            features=self.args.features,
            scale=True,
            target=self.args.target,
            noise_level=noise_level
        )

        # 3. 构造 DataLoader
        use_persistent = (self.args.num_workers > 0)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
            persistent_workers=use_persistent,
            pin_memory=self.args.use_gpu
        )

        return data_set, data_loader

    def _build_model(self):
        # 1.临时获取 Dataset 以提取统计参数
        train_data, _ = self._get_data(flag='train')

        # 2. 直接从训练集的 Scaler 中提取参数
        # PhysFormerDataset 已经有一个 get_scaler_params 方法，可以复用
        scaler_params = train_data.get_scaler_params()

        # 检查是否获取成功
        if scaler_params['mean'] is not None:
            # 假设前3列分别是 [Load, PV, Wind]，这取决于你的CSV列顺序
            # 这里的切片操作就是我们之前说的"提取/计算"
            target_mean = scaler_params['mean'][:3]
            target_std = scaler_params['std'][:3]

            # 天气参数 (Temp, Irr, Speed)
            weather_mean = scaler_params['weather_mean']
            weather_std = scaler_params['weather_std']
        else:
            # 兜底逻辑（如果没开启归一化）
            target_mean, target_std = None, None
            weather_mean, weather_std = None, None

        model = PhysFormer(
            enc_in=self.args.enc_in,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            attn=self.args.attn,
            embed=self.args.embed,
            freq=self.args.freq,
            activation=self.args.activation,
            distil=self.args.distil,
            weather_mean=weather_mean,
            weather_std=weather_std,
            target_mean=target_mean,
            target_std=target_std,
            device=self.device,
            use_rope=self.args.use_rope,
            rope_base=self.args.rope_base,
        )

        return model

    def _get_curriculum_ratio(self, epoch):
        """
        计算当前 Epoch 的课程进度 (0.0 -> 1.0)
        策略: Sigmoid-like Growth
        - Epoch 0-5: 0.0 (Warmup Pinball only)
        - Epoch 5-20: 0.0 -> 1.0 (Gradual Injection)
        - Epoch 20+: 1.0 (Full Constraints)
        """
        # 可以设为超参数
        start_epoch = 5
        ramp_epochs = 15

        if epoch < start_epoch:
            return 0.0
        if epoch >= start_epoch + ramp_epochs:
            return 1.0

        # Sigmoid 映射
        progress = (epoch - start_epoch) / ramp_epochs  # 0 -> 1
        # 映射到 -6 到 6 区间进行 sigmoid，产生平滑的 S 曲线
        import math
        x = (progress - 0.5) * 12
        ratio = 1 / (1 + math.exp(-x))
        return ratio

    def _select_criterion(self):
        # 1. 获取训练集
        if not hasattr(self, 'train_dataset'):
            # 注意：这里我们获取 dataset 对象，而不仅仅是 loader
            self.train_dataset, _ = self._get_data(flag='train')

        # 2. 从 Dataset 中直接获取物理统计量 [NEW]
        # PhysFormerDataset 现在负责计算这些值
        phys_stats = self.train_dataset.get_physical_stats()

        real_means = phys_stats['means']
        real_stds = phys_stats['stds']
        real_ramp_limits = phys_stats['ramp_limits']

        # 日志记录
        self.logger.info(f">>> [Auto-Stat] Loaded Real Means: {real_means}")
        self.logger.info(f">>> [Auto-Stat] Loaded Real Stds : {real_stds}")
        self.logger.info(f">>> [Auto-Stat] Calculated Ramp Limits: {real_ramp_limits}")

        # 保存到 self 供后续使用
        self.train_means = real_means
        self.train_stds = real_stds
        self.train_ramp_limits = real_ramp_limits

        # 4. 实例化 Micro-Level Base Loss
        base_criterion = PhysAwareBaseLoss(
            device=self.device,
            means=self.train_means,
            stds=self.train_stds,
            ramp_limits=self.train_ramp_limits
        )

        # 5. 实例化 Macro-Level Wrapper (ProductionReadyPhysLoss)
        # 注意：这里我们使用你新定义的类名，假设你文件里叫 PhysLoss
        criterion = PhysLoss(
            base_loss_module=base_criterion,
            device=self.device,
            warmup_batches=50,
            ema_decay=0.9
        )

        return criterion

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-5
        params = list(self.model.parameters())

        model_optim = optim.AdamW(
            params,
            lr=self.args.learning_rate,
            weight_decay=wd
        )
        return model_optim

    def _process_one_batch(self, batch_data, criterion=None, phase='train', return_gates=False, curriculum_ratio=0.0):
        # 1. 解包数据并转移到 GPU
        batch_stat, batch_weather_hist, batch_weather_future, batch_y, batch_x_mark, batch_y_mark = batch_data
        # 2. 转移到 GPU
        batch_stat = batch_stat.float().to(self.device)  # [B, Seq, 3]
        batch_weather_hist = batch_weather_hist.float().to(self.device)  # [B, Seq, 3]
        batch_weather_future = batch_weather_future.float().to(self.device)  # [B, Pred, 3]
        batch_y = batch_y.float().to(self.device)  # [B, Pred, 3] (Label)
        batch_x_mark = batch_x_mark.float().to(self.device)  # [B, Seq, 8]
        # batch_y_mark = batch_y_mark.float().to(self.device)     # Encoder-Only模型不需要未来时间戳

        # 3. 准备 Ground Truth
        # Dataset 返回的 batch_y 已经是未来真值 [B, Pred, 3]
        batch_y_true = batch_y

        # 4. 前向传播
        # 确保使用正确的 Context (Train/Val/Test)
        context = torch.cuda.amp.autocast(enabled=self.args.use_amp) if self.args.use_gpu else torch.no_grad()
        if phase in ['val', 'test']:
            # 验证时通常不需要 autocast 或保持一致，这里简单起见跟随配置
            pass

        with context:
            if return_gates:
                outputs, gates = self.model(
                    x_stat=batch_stat,
                    x_weather_hist=batch_weather_hist,
                    x_weather_future=batch_weather_future,
                    x_mark_enc=batch_x_mark,
                    return_gates=True
                )
            else:
                outputs = self.model(
                    x_stat=batch_stat,
                    x_weather_hist=batch_weather_hist,
                    x_weather_future=batch_weather_future,
                    x_mark_enc=batch_x_mark,
                    return_gates=False
                )
                gates = None

            total_loss = None
            loss_dict = {}

            if criterion is not None:
                # 计算 Loss (MSE + Phys)
                loss_main, loss_dict = criterion(outputs, batch_y_true, curriculum_ratio=curriculum_ratio)

                # 加上 Gate Regularization
                if gates is not None and 'gate_reg_loss' in gates:
                    reg_loss = gates['gate_reg_loss']
                    if isinstance(reg_loss, torch.Tensor):
                        loss_main += reg_loss
                        loss_dict['gate_reg'] = reg_loss.item()

                total_loss = loss_main

        if return_gates:
            return outputs, batch_y_true, total_loss, loss_dict, gates
        else:
            return outputs, batch_y_true, total_loss, loss_dict

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, self.args.checkpoint_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # 1. 初始化 Loss (获取 log_vars 参数)
        self.criterion = self._select_criterion()
        self.criterion.to(self.device)

        # 2. 再初始化 Optimizer (传入 Model + Loss 的参数)
        model_optim = self._select_optimizer()

        # 3. 早停机制 (监控 物理MAE)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

        # 4. 学习率调度器: 余弦退火
        scheduler = CosineAnnealingLR(
            model_optim,
            T_max=self.args.train_epochs,
            eta_min=1e-6
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

        # 全局步数计数器
        global_step = 0

        for epoch in range(self.args.train_epochs):
            self.model.train()
            self.criterion.train()

            # 获取当前 Epoch 的课程 Ratio
            curr_ratio = self._get_curriculum_ratio(epoch)

            train_loss_log = []
            debug_log = {'scale': []}

            with tqdm(total=len(train_loader),
                      desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                      mininterval=0.3,
                      leave=False,
                      ncols=100) as pbar:
                for i, batch_data in enumerate(train_loader):
                    model_optim.zero_grad()

                    # Train 阶段我们还需要加上惯性正则化 Loss，所以我们在外层加
                    outputs, batch_y_true, loss_main, loss_dict = self._process_one_batch(
                        batch_data, self.criterion, phase='train', return_gates=False,
                        curriculum_ratio=curr_ratio
                    )

                    # Backward
                    scaler.scale(loss_main).backward()
                    scaler.unscale_(model_optim)

                    # 在 Optimizer Step 之前记录梯度
                    if i % 50 == 0:  # 每 50 个 batch 记录一次梯度，避免拖慢速度
                        self._log_gradient_norms(global_step)

                    # 动态梯度裁剪
                    clip_norm = 0.5 if 0.1 < curr_ratio < 0.9 else 1.0
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                    scaler.step(model_optim)
                    scaler.update()

                    # [新增] 记录 Loss 和 物理参数
                    if i % 20 == 0:  # 每 20 个 batch 记录一次 Loss
                        # 记录主 Loss
                        self.writer.add_scalar('Loss/Total_Train', loss_main.item(), global_step)
                        self.writer.add_scalar('Training/Curriculum_Ratio', curr_ratio, global_step)

                        # 记录分项物理 Loss (从 loss_dict 中取)
                        if loss_dict:
                            if 'mse' in loss_dict:
                                self.writer.add_scalar('Loss_Components/MSE', loss_dict['mse'], global_step)
                            for k, v in loss_dict.items():
                                if k in ['net', 'energy', 'deriv', 'bvr', 'rvr']:
                                    self.writer.add_scalar(f'Loss_Components/{k}', v, global_step)
                            if 'scale' in loss_dict:
                                self.writer.add_scalar('Training/Balancing_Scale', loss_dict['scale'], global_step)

                        # 记录显式物理参数
                        self._log_physical_params(global_step)

                    global_step += 1

                    if 'scale' in loss_dict:
                        debug_log['scale'].append(loss_dict['scale'])

                    train_loss_log.append(loss_main.item())
                    pbar.update(1)

            # --- C. 验证与早停 ---
            # 这里的 vali_mae 是反归一化后的真实物理误差 (MW)
            vali_loss, vali_mae, avg_gates = self.vali(val_data, val_loader, self.criterion, epoch=epoch)

            # 记录gate历史
            self.gate_history['epoch'].append(epoch + 1)
            self.gate_history['load'].append(avg_gates['load'])
            self.gate_history['pv'].append(avg_gates['pv'])
            self.gate_history['wind'].append(avg_gates['wind'])
            self.gate_history['info_gain'].append(avg_gates['info_gain'])

            # 调度器步进 (WarmRestarts 是按 epoch 更新的)
            scheduler.step()

            # Logging
            avg_scale = np.mean(debug_log['scale']) if debug_log['scale'] else 1.0

            self.logger.info(
                f"Epoch {epoch + 1} | Ratio: {curr_ratio:.3f} | "
                f"Balancing Scale: {avg_scale:.2f}"
            )

            # 打印各项物理Loss的原始值，而非权重
            phys_terms = ['net', 'energy', 'deriv', 'bvr']
            phys_str = " | ".join([f"{k}:{loss_dict.get(k, 0):.4f}" for k in phys_terms])
            self.logger.info(f"  >> [Phys Losses] {phys_str}")

            self.logger.info(
                f"  >> [Gates] Load: {avg_gates['load']:.3f} | PV: {avg_gates['pv']:.3f} | "
                f"Wind: {avg_gates['wind']:.3f}"
            )

            # 每个 Epoch 结束记录验证集 MAE
            self.writer.add_scalar('Loss/Val_MAE', vali_mae, epoch)
            # 记录平均 Gate 值
            self.writer.add_scalar('Gates/Avg_Load', avg_gates['load'], epoch)
            self.writer.add_scalar('Gates/Avg_PV', avg_gates['pv'], epoch)
            self.writer.add_scalar('Gates/Avg_Wind', avg_gates['wind'], epoch)

            early_stopping(vali_mae, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        # 训练结束后
        # 保存gate历史
        gate_save_path = os.path.join(path, 'gate_history.npy')
        np.save(gate_save_path, self.gate_history)
        self.logger.info(f"Gate history saved to {gate_save_path}")

        # Load Best Model
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion, epoch=0):
        self.model.eval()
        self.criterion.eval()
        total_loss = []
        total_metrics = {'mae': []}
        batch_gates = {'load': [], 'pv': [], 'wind': [], 'info_gain': []}

        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                outputs, batch_y_true, loss, loss_dict, gates= self._process_one_batch(
                    batch_data, criterion, phase='val', return_gates=True,
                    curriculum_ratio=1.0    # 使用完整物理约束
                )
                total_loss.append(loss.item())
                if 'mae' in loss_dict:
                    total_metrics['mae'].append(loss_dict['mae'])

                if gates is not None:
                    for k in batch_gates.keys():
                        if k in gates:
                            batch_gates[k].append(gates[k])

                # 仅在每个 Epoch 的第一个 Batch 进行绘图
                if i == 0:
                    fig = self._plot_prediction(outputs, batch_y_true, vali_data)
                    # 写入 TensorBoard
                    # tag 格式：Val_Prediction/Epoch
                    self.writer.add_figure('Validation/Prediction_Sample', fig, global_step=epoch)
                    plt.close(fig)  # 重要！画完记得关闭，否则内存泄漏

        avg_loss = np.average(total_loss)
        avg_mae = np.mean(total_metrics['mae']) if len(total_metrics['mae']) > 0 else 0.0
        avg_gates = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in batch_gates.items()}

        self.model.train()
        self.criterion.train()

        return avg_loss, avg_mae, avg_gates

    def test(self, setting, load=True):
        test_data, test_loader = self._get_data(flag='test')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        # 检查criterion是否存在
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.logger.warning("Warning: Criterion not found. Rebuilding from training stats...")
            self.criterion = self._select_criterion()

        self.model.eval()
        self.criterion.eval()

        preds, trues = [], []
        detailed_gates = {'load': [], 'pv': [], 'wind': [], 'pv_true': [], 'timestamps': []}
        phys_metrics = {'mse': [], 'mae': [], 'net': [], 'deriv': [], 'energy': [], 'dir': [], 'bvr': [], 'rvr': []}
        vis_data = {'gate_pv': [], 'gate_wind': [], 'irr': [], 'speed': []}

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                outputs, batch_y_true, _, loss_dict, gates = self._process_one_batch(
                    batch_data, self.criterion, phase='test', return_gates=True,
                    curriculum_ratio=1.0
                )

                if gates:
                    detailed_gates['load'].append(gates.get('load', 0))
                    detailed_gates['pv'].append(gates.get('pv', 0))
                    detailed_gates['wind'].append(gates.get('wind', 0))
                detailed_gates['pv_true'].append(batch_y_true[:, :, 1].mean().item())
                detailed_gates['timestamps'].append(i)

                # 只取第一个样本 (Sample 0) 避免平均化模糊
                if i < 5:
                    if 'pv_seq_batch' in gates:
                        vis_data['gate_pv'].append(gates['pv_seq_batch'][0])
                        vis_data['gate_wind'].append(gates['wind_seq_batch'][0])

                        vis_data['irr'].append(gates['irr_seq_batch'][0])
                        vis_data['speed'].append(gates['speed_seq_batch'][0])

                for k in phys_metrics.keys():
                    if k in loss_dict:
                        phys_metrics[k].append(loss_dict[k])

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y_true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # --- 反归一化逻辑 ---
        if test_data.scale and test_data.scaler is not None:
            # 展平 [N*P, 3]
            shape_orig = preds.shape
            preds_2d = preds.reshape(-1, shape_orig[-1])
            trues_2d = trues.reshape(-1, shape_orig[-1])

            # 使用 Dataset 中自定义的 smart inverse_transform (支持自动补全列)
            preds_rescaled = test_data.inverse_transform(preds_2d)
            trues_rescaled = test_data.inverse_transform(trues_2d)

            # 恢复形状
            preds = preds_rescaled.reshape(shape_orig)
            trues = trues_rescaled.reshape(shape_orig)

        metrics_result = metric(preds, trues)
        avg_phys_metrics = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in phys_metrics.items()}

        print("\n" + "=" * 60)
        print("  VPP Physics Compliance Report (Test Set)")
        print("=" * 60)
        for k, v in avg_phys_metrics.items():
            print(f"  {k.upper():<20} : {v:.6f}")
        print("=" * 60 + "\n")

        path = os.path.join(self.args.checkpoints, self.args.checkpoint_name)
        if not os.path.exists(path): os.makedirs(path)
        folder_path = path + '/'

        np.save(os.path.join(folder_path, 'vis_gate_pv.npy'), np.array(vis_data['gate_pv']))
        np.save(os.path.join(folder_path, 'vis_gate_wind.npy'), np.array(vis_data['gate_wind']))
        np.save(os.path.join(folder_path, 'vis_irr.npy'), np.array(vis_data['irr']))
        np.save(os.path.join(folder_path, 'vis_speed.npy'), np.array(vis_data['speed']))

        np.save(os.path.join(folder_path, 'gate_details.npy'), detailed_gates)
        np.save(os.path.join(folder_path, 'phys_metrics.npy'), avg_phys_metrics)
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array(metrics_result))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return preds, trues

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