import math

import torch
import torch.optim as optim
import numpy as np
import os
import warnings
import logging
import matplotlib.pyplot as plt
import json

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .base import EarlyStopping, ForecastExperiment
from ..data.data_factory import PhysFormerDataset
from ..models import PhysFormer
from ..utils.losses import PhysAwareBaseLoss, PhysLoss
from ..utils.metrics import metric, NRMSE_Channel_Avg, PhysicsComplianceMetrics

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

class Exp_PhysFormer(ForecastExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._build_model().to(self.device)

        self.gate_history = {
            'epoch': [], 'load': [], 'pv': [], 'wind': [],
            'info_gain': [], 'volatility': [], 'weather': []
        }

        # 初始化 TensorBoard Writer
        # 日志保存在 checkpoints/你的实验名/tensorboard 目录下
        tb_dir = self.run_dir / 'tensorboard'
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
            noise_level = 0.01
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
            noise_level=noise_level,
            time_col=getattr(self.args, 'time_col', 'date'),
            id_col=getattr(self.args, 'id_col', None),
            region_col=getattr(self.args, 'region_col', None),
            target_cols=getattr(self.args, 'target_cols', None),
            covariate_cols=getattr(self.args, 'covariate_cols', None),
            task_mode=getattr(self.args, 'task_mode', 'component_multitask')
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
            # --- ABLATION FLAGS ---
            ablation_no_phys_stream=getattr(self.args, 'ablation_no_phys_stream', False),
            ablation_no_pgcc=getattr(self.args, 'ablation_no_pgcc', False),
            ablation_no_future_glu=getattr(self.args, 'ablation_no_future_glu', False),
            ablation_no_curriculum=getattr(self.args, 'ablation_no_curriculum', False),
            ablation_fixed_phys=getattr(self.args, 'ablation_fixed_phys', False),
            ablation_no_gate_reg=getattr(self.args, 'ablation_no_gate_reg', False),
        )

        return model

    def _get_curriculum_ratio(self, epoch):
        """
        分层课程学习：返回每个物理约束的独立权重字典。
        三阶段策略：
        - 阶段1 (epoch 0-5): 纯数据驱动，所有权重为0
        - 阶段2 (epoch 5-15): 逐步引入网络约束（net 从0→1.0）
        - 阶段3 (epoch 15-30): 逐步引入能量与导数约束（energy 0→0.5，deriv 0→0.3）

        硬约束（bvr, rvr）从阶段2开始与net同步介入。
        方向约束（dir）从阶段2开始与net同步介入。

        返回字典格式：{'net': w1, 'energy': w2, 'deriv': w3, 'bvr': w4, 'rvr': w5, 'dir': w6}
        curriculum 权重归一化到 [0, 1] 范围（代表"注入多少比例"）
        各物理项的相对重要性完全由 PhysLoss.sub_weights 承载
        两者职责分离，互不重叠。
        """
        # 阶段定义 (配合 warmup_epochs = 10 延长平稳期)
        stage1_end = 10
        stage2_end = 15
        stage3_end = 35

        # 最终目标权重（基于指导文档和现有实现）
        target_weights = {
            'net': 1.0,
            'energy': 1.0,
            'deriv': 1.0,
            'bvr': 1.0,
            'rvr': 1.0,
            'dir': 1.0,
        }

        # --- ABLATION: w/o Curriculum ---
        if getattr(self.args, 'ablation_no_curriculum', False):
            return target_weights

        # 初始化权重字典
        weights = {k: 0.0 for k in target_weights.keys()}

        if epoch < stage1_end:
            # 阶段1：纯数据驱动，所有权重为0
            return weights

        if epoch >= stage3_end:
            return target_weights

        elif epoch >= stage2_end:
            # 阶段3：net/bvr/rvr/dir 已满权重，energy/deriv 渐入
            stage3_progress = (epoch - stage2_end) / (stage3_end - stage2_end)
            x = (stage3_progress - 0.5) * 12
            ratio3 = 1 / (1 + math.exp(-x))

            weights['net'] = target_weights['net']
            weights['bvr'] = target_weights['bvr']
            weights['rvr'] = target_weights['rvr']
            weights['dir'] = target_weights['dir']
            weights['energy'] = target_weights['energy'] * ratio3  # 0 → 0.5
            weights['deriv'] = target_weights['deriv'] * ratio3  # 0 → 0.3
            return weights

        else:
            # 阶段2逻辑不变（epoch 10-15）
            stage2_progress = (epoch - stage1_end) / (stage2_end - stage1_end)
            x = (stage2_progress - 0.5) * 12
            ratio = 1 / (1 + math.exp(-x))
            weights['net'] = target_weights['net'] * ratio
            weights['bvr'] = target_weights['bvr'] * ratio
            weights['rvr'] = target_weights['rvr'] * ratio
            weights['dir'] = target_weights['dir'] * ratio
            return weights

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
            ema_decay=0.95
        )

        return criterion

    def _select_optimizer(self):
        base_lr = self.args.learning_rate
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-5

        # 参数分类
        phys_params, gate_params, stat_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'phys_layer.' in name:           # 精确匹配 model.phys_layer.xxx
                phys_params.append(param)
            elif 'causal_coupling.' in name:    # 精确匹配耦合模块所有参数
                gate_params.append(param)
            else:
                stat_params.append(param)

        # 打印参数分组信息（调试用）
        print(f">>> [Layer-wise LR] Phys: {len(phys_params)} params, Gate: {len(gate_params)} params, Stat: {len(stat_params)} params")

        param_groups = [
            {'params': phys_params, 'lr': base_lr * 0.1, 'name': 'phys_params'},
            {'params': gate_params, 'lr': base_lr * 0.5, 'name': 'gate_params'},
            {'params': stat_params, 'lr': base_lr * 1.0, 'name': 'stat_params'},
        ]

        # Enable fused AdamW for massive multi-tensor update speedup on modern GPUs 
        if self.device.type == 'cuda' and int(torch.__version__.split('.')[0]) >= 2:
            model_optim = optim.AdamW(param_groups, weight_decay=wd, fused=True)
            self.logger.info(">>> [Optimizer] Using Fused AdamW")
        else:
            model_optim = optim.AdamW(param_groups, weight_decay=wd)

        for p_group in param_groups:
            sample_names = [n for n, p in self.model.named_parameters()
                            if any(p is pp for pp in p_group['params'])][:3]
            print(f">>> Group '{p_group['name']}' lr={p_group['lr']:.2e}, "
                  f"sample params: {sample_names}")

        return model_optim

    def _process_one_batch(self, batch_data, criterion=None, phase='train',
                           return_gates=False, curriculum_ratio=0.0, current_prior_weight=0.0):
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

        # 提取 network 的 curriculum ratio
        progress_ratio = curriculum_ratio.get('net', 0.0) if isinstance(curriculum_ratio, dict) else curriculum_ratio
        # BUGFIX: 不要让 alpha 衰减到 0.0。强制保留最低 10% 的物理硬约束介入作为全局正则化，
        # 这也是修复前 Val NRMSE 能达到 0.036 的关键泛化兜底。
        alpha = max(0.1, 1.0 - progress_ratio)

        # 4. 前向传播
        # 确保使用正确的 Context (Train/Val/Test)
        # BUGFIX: 验证和测试阶段强制关闭 autocast 以避免更易触发的 fp16 溢出
        is_train = (phase == 'train')
        context = torch.cuda.amp.autocast(enabled=(self.args.use_amp and is_train)) if self.args.use_gpu else torch.no_grad()

        with context:
            # 移除 detect_anomaly 的不必要开销，除非强制 debug
            if return_gates:
                outputs, reg_loss, gates = self.model(
                    x_stat=batch_stat,
                    x_weather_hist=batch_weather_hist,
                    x_weather_future=batch_weather_future,
                    x_mark_enc=batch_x_mark,
                    alpha=alpha,
                    return_gates=True
                )
            else:
                outputs, reg_loss = self.model(
                    x_stat=batch_stat,
                    x_weather_hist=batch_weather_hist,
                    x_weather_future=batch_weather_future,
                    x_mark_enc=batch_x_mark,
                    alpha=alpha,
                    return_gates=False
                )
                gates = None

        total_loss = None
        loss_dict = {}

        if criterion is not None:
            # [PERFORMANCE FIX] 将 MSE 等海量运算相关的基础 Loss 内置到 AMP 回收利用算力
            # 这里重置计算栈重新回到 autocast，充分发挥 TensorCore 在处理矩阵大加减运算时的优势
            with context:
                # 只在需要的情况下强转保证稳定性 (由具体的 Criterion / Layer 内部通过 autocast(False) 控制)
                loss_main, loss_dict = criterion(outputs, batch_y_true, curriculum_weights=curriculum_ratio)

                # 将门控的响应度正则化 Loss 叠加进计算图
                if isinstance(reg_loss, torch.Tensor) and reg_loss.requires_grad:
                    loss_main += reg_loss
                    loss_dict['gate_reg'] = reg_loss.item()

                # 加上物理参数先验正则化（仅在训练阶段）
                # --- ABLATION: w/o Physics Stream ---
                ablation_no_phys = getattr(self.args, 'ablation_no_phys_stream', False) or getattr(self.model, 'ablation_no_phys_stream', False)
                if phase == 'train' and hasattr(self.model, 'phys_layer') and current_prior_weight > 0 and not ablation_no_phys:
                    prior_loss, prior_dict = self.model.phys_layer.get_physics_prior_loss(prior_weight=current_prior_weight)
                    if isinstance(prior_loss, torch.Tensor):
                        loss_main += prior_loss
                        loss_dict['physics_prior'] = prior_loss.item()
                        # 记录各项先验损失用于监控
                        for k, v in prior_dict.items():
                            loss_dict[f'prior_{k}'] = v

                total_loss = loss_main

        if return_gates:
            return outputs, batch_y_true, total_loss, loss_dict, gates
        else:
            return outputs, batch_y_true, total_loss, loss_dict

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        path = str(self.run_dir)
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
            T_max=max(1, int(self.args.train_epochs * 0.7)),  # 70%处降至最低LR，匹配early stop节奏
            eta_min=1e-6
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

        # 全局步数计数器与断点恢复逻辑
        global_step = 0
        start_epoch = 0

        if getattr(self.args, 'debug_nan', False):
            torch.autograd.set_detect_anomaly(True)
            self.logger.info(">>> [DEBUG] PyTorch Anomaly Detection Enabled! <<<")

        if getattr(self.args, 'resume', False) and os.path.exists(path + '/checkpoint.pth'):
            self.logger.info(f">>> [RESUME] Loading checkpoint from {path} <<<")
            self.model.load_state_dict(torch.load(path + '/checkpoint.pth', weights_only=False))
            state_path = path + '/training_state.pth'
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=self.device, weights_only=False)
                model_optim.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                scaler.load_state_dict(state['scaler'])
                start_epoch = state['epoch'] + 1
                global_step = state['global_step']
                early_stopping.best_score = state['best_score']
                early_stopping.val_loss_min = state['val_loss_min']
                early_stopping.counter = state['counter']
                self.logger.info(f">>> [RESUME] Successfully resumed from Epoch {state['epoch']} (Next: {start_epoch}) <<<")
            else:
                self.logger.warning(">>> [RESUME WARNING] training_state.pth not found! Resuming model weights ONLY. <<<")

        for epoch in range(start_epoch, self.args.train_epochs):
            self.model.train()
            self.criterion.train()

            # 物理参数的冻结与解冻机制 ---
            # 设定预热期为 10 个 epoch，让 Transformer 侧有充足的初期收敛时间，降低物理极早期介入的抖动
            warmup_epochs = 10

            # --- ABLATION: Fixed Thresholds OR w/o Curriculum ---
            if getattr(self.args, 'ablation_fixed_phys', False):
                # Always freeze physical layer if ablation_fixed_phys is True
                if hasattr(self.model, 'phys_layer'):
                    for param in self.model.phys_layer.parameters():
                        param.requires_grad = False
                current_prior_weight = 0.0
                if epoch == 0:
                    self.logger.info(">>> [PhysLayer Control] ABLATION_FIXED_PHYS: Physics parameters FROZEN for all epochs.")
            elif getattr(self.args, 'ablation_no_curriculum', False):
                # Never freeze physics layer if ablation_no_curriculum is True
                if hasattr(self.model, 'phys_layer'):
                    for param in self.model.phys_layer.parameters():
                        param.requires_grad = True
                current_prior_weight = getattr(self.args, 'physics_prior_weight', 0.1)
                if epoch == 0:
                    self.logger.info(f">>> [PhysLayer Control] ABLATION_NO_CURRICULUM: Physics parameters UNFROZEN from Epoch 1. Prior weight: {current_prior_weight}")
            else:
                if epoch < warmup_epochs:
                    # 阶段一：冻结物理层，强制 Transformer 适应标准物理先验
                    if hasattr(self.model, 'phys_layer'):
                        for param in self.model.phys_layer.parameters():
                            param.requires_grad = False
                    current_prior_weight = 0.0  # 冻结状态不需要计算先验 Loss
                    if epoch == 0:
                        self.logger.info(f">>> [PhysLayer Control] Epoch 1-{warmup_epochs}: Physics parameters FROZEN.")
                else:
                    # 阶段二：解冻物理层，允许数据驱动的微调，并施加先验约束防止跑偏
                    if hasattr(self.model, 'phys_layer'):
                        for param in self.model.phys_layer.parameters():
                            param.requires_grad = True
                    # prior_weight 随训练进度线性衰减：初期强约束防跑偏，后期放松允许精细适应
                    base_prior_weight = getattr(self.args, 'physics_prior_weight', 0.1)
                    decay_ratio = max(0.2, 1.0 - (epoch - warmup_epochs) / (self.args.train_epochs - warmup_epochs))
                    current_prior_weight = base_prior_weight * decay_ratio
                    if epoch == warmup_epochs:
                        self.logger.info(
                            f">>> [PhysLayer Control] Epoch {epoch + 1}+: Physics parameters UNFROZEN. Prior weight: {current_prior_weight:.4f} (decay enabled)")
            # ----------------------------------------------------

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
                    model_optim.zero_grad(set_to_none=True)

                    # Train 阶段我们还需要加上惯性正则化 Loss，所以我们在外层加
                    outputs, batch_y_true, loss_main, loss_dict = self._process_one_batch(
                        batch_data, self.criterion, phase='train', return_gates=False,
                        curriculum_ratio=curr_ratio, current_prior_weight=current_prior_weight
                    )

                    if getattr(self.args, 'debug_nan', False) and not torch.isfinite(loss_main):
                        self.logger.error(f"[DEBUG] NaN Loss detected at Epoch {epoch}, Batch {i}")
                        self.logger.error(f"Loss dict: {loss_dict}")
                        raise ValueError("NaN Loss detected! Please check forward pass logic.")

                    # Backward
                    scaler.scale(loss_main).backward()
                    scaler.unscale_(model_optim)

                    # 在 Optimizer Step 之前记录梯度
                    if i % 50 == 0:  # 每 50 个 batch 记录一次梯度，避免拖慢速度
                        self._log_gradient_norms(global_step)

                    # 统一梯度裁剪：不再在阶段2收紧clip，避免物理约束梯度被卡死
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()

                    # 记录 Loss 和 物理参数
                    if i % 20 == 0:  # 每 20 个 batch 记录一次 Loss
                        # 记录主 Loss
                        self.writer.add_scalar('Loss/Total_Train', loss_main.item(), global_step)
                        # 记录课程权重（curr_ratio现在是字典）
                        if isinstance(curr_ratio, dict):
                            for key, value in curr_ratio.items():
                                self.writer.add_scalar(f'Curriculum/{key}', value, global_step)
                            # 同时记录平均权重以便监控
                            avg_ratio = sum(curr_ratio.values()) / len(curr_ratio) if curr_ratio else 0.0
                            self.writer.add_scalar('Curriculum/Average', avg_ratio, global_step)
                        else:
                            # 向后兼容：如果是标量，按原样记录
                            self.writer.add_scalar('Training/Curriculum_Ratio', curr_ratio, global_step)

                        # 记录分项物理 Loss (从 loss_dict 中取)
                        if loss_dict:
                            if 'mse' in loss_dict:
                                self.writer.add_scalar('Loss_Components/MSE', loss_dict['mse'], global_step)
                            for k, v in loss_dict.items():
                                if k in ['net', 'energy', 'deriv', 'bvr', 'rvr']:
                                    self.writer.add_scalar(f'Loss_Components/{k}', v, global_step)
                                elif k == 'physics_prior':
                                    self.writer.add_scalar('Loss_Components/Physics_Prior', v, global_step)
                                elif k.startswith('prior_'):
                                    # 记录先验损失的各个组成部分
                                    self.writer.add_scalar(f'Physics_Prior/{k[6:]}', v, global_step)
                                elif k.startswith('curriculum_'):
                                    # 记录课程权重
                                    self.writer.add_scalar(f'Curriculum/{k[11:]}', v, global_step)
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
            # 这里的 vali_nrmse 是消除负荷极大值会带来的收敛影响
            vali_loss, vali_nrmse, avg_gates = self.vali(val_data, val_loader, self.criterion, epoch=epoch)

            # 记录gate历史
            self.gate_history['epoch'].append(epoch + 1)
            self.gate_history['load'].append(avg_gates['load'])
            self.gate_history['pv'].append(avg_gates['pv'])
            self.gate_history['wind'].append(avg_gates['wind'])
            self.gate_history['info_gain'].append(avg_gates['info_gain'])
            self.gate_history['volatility'].append(avg_gates.get('volatility', 0.0))
            self.gate_history['weather'].append(avg_gates.get('weather_gate', 0.0))

            # 调度器步进 (WarmRestarts 是按 epoch 更新的)
            scheduler.step()

            # Logging
            avg_scale = np.mean(debug_log['scale']) if debug_log['scale'] else 1.0

            # 格式化课程权重信息
            if isinstance(curr_ratio, dict):
                # 计算平均权重
                avg_ratio = sum(curr_ratio.values()) / len(curr_ratio) if curr_ratio else 0.0
                # 提取关键权重用于显示
                key_ratios = []
                for key in ['net', 'energy', 'deriv']:
                    if key in curr_ratio:
                        key_ratios.append(f"{key}:{curr_ratio[key]:.3f}")
                ratio_str = f"Avg:{avg_ratio:.3f} ({', '.join(key_ratios)})"
            else:
                ratio_str = f"{curr_ratio:.3f}"

            self.logger.info(
                f"Epoch {epoch + 1} | Ratio: {ratio_str} | "
                f"Balancing Scale: {avg_scale:.2f}"
            )

            # 打印全部物理指标（两行）
            self.logger.info(
                f"  >> [Fit]  MSE:{loss_dict.get('mse', 0):.4f} | MAE:{loss_dict.get('mae', 0):.4f}"
            )
            phys_terms = ['net', 'energy', 'deriv', 'dir', 'bvr', 'rvr', 'physics_prior']
            phys_str = " | ".join([f"{k}:{loss_dict.get(k, 0):.4f}" for k in phys_terms])
            self.logger.info(f"  >> [Phys] {phys_str}")

            # 每个 Epoch 结束记录验证集 NRMSE
            self.writer.add_scalar('Loss/Val_NRMSE', vali_nrmse, epoch)
            # 记录平均 Gate 值
            self.writer.add_scalar('Gates/Avg_Load', avg_gates['load'], epoch)
            self.writer.add_scalar('Gates/Avg_PV', avg_gates['pv'], epoch)
            self.writer.add_scalar('Gates/Avg_Wind', avg_gates['wind'], epoch)
            self.writer.add_scalar('Gates/Avg_Weather_Injection', avg_gates.get('weather_gate', 0.0), epoch)

            early_stopping(vali_nrmse, self.model, path, model_optim, scheduler, scaler, epoch, global_step)
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
        self.model.load_state_dict(torch.load(best_model_path, weights_only=False))
        return self.model

    def vali(self, vali_data, vali_loader, criterion, epoch=0):
        self.model.eval()
        self.criterion.eval()
        total_loss = []

        # 收集预测值用于计算 NRMSE
        preds = []
        trues = []

        batch_gates = {
            'load': [], 'pv': [], 'wind': [],
            'info_gain': [], 'volatility': [], 'weather_gate': []
        }

        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                outputs, batch_y_true, loss, loss_dict, gates = self._process_one_batch(
                    batch_data, criterion, phase='val', return_gates=True,
                    curriculum_ratio=1.0, current_prior_weight=0.0
                )
                total_loss.append(loss.item())

                # 收集每一批次的预测和真实值
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y_true.detach().cpu().numpy())

                if gates is not None:
                    for k in batch_gates.keys():
                        if k in gates:
                            batch_gates[k].append(gates[k])

                if i == 0:
                    fig = self._plot_prediction(outputs, batch_y_true, vali_data)
                    self.writer.add_figure('Validation/Prediction_Sample', fig, global_step=epoch)
                    plt.close(fig)

        # 拼接全部样本
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # 反归一化到真实物理单位 (极度重要，否则 NRMSE 的分母 range 无物理意义)
        if vali_data.scale and vali_data.scaler is not None:
            shape_orig = preds.shape
            preds_2d = preds.reshape(-1, shape_orig[-1])
            trues_2d = trues.reshape(-1, shape_orig[-1])
            preds = vali_data.inverse_transform(preds_2d).reshape(shape_orig)
            trues = vali_data.inverse_transform(trues_2d).reshape(shape_orig)

        # 计算消除了量纲影响的平均 NRMSE
        avg_nrmse = NRMSE_Channel_Avg(preds, trues)
        avg_loss = np.average(total_loss)
        avg_gates = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in batch_gates.items()}

        self.model.train()
        self.criterion.train()

        # 返回 avg_nrmse 作为早停依据
        return avg_loss, avg_nrmse, avg_gates

    def test(self, setting=None, load=True, return_preds=False, extreme_scenario_test=False):
        test_data, test_loader = self._get_data(flag='test')

        if load:
            best_model_path = str(self.checkpoint_path())
            self.model.load_state_dict(torch.load(best_model_path, weights_only=False))

        # 检查criterion是否存在
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.logger.warning("Warning: Criterion not found. Rebuilding from training stats...")
            self.criterion = self._select_criterion()

        actual_ramp_limits = self.train_ramp_limits

        self.model.eval()
        self.criterion.eval()

        preds, trues = [], []
        weather_data_list = []  # 收集天气数据用于极端场景测试
        detailed_gates = {'load': [], 'pv': [], 'wind': [], 'pv_true': [], 'timestamps': []}
        phys_metrics = {'mse': [], 'mae': [], 'net': [], 'deriv': [], 'energy': [], 'dir': [], 'bvr': [], 'rvr': []}
        vis_data = {'gate_pv': [], 'gate_wind': [], 'irr': [], 'speed': []}

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                # 解包数据以获取天气数据
                batch_stat, batch_weather_hist, batch_weather_future, batch_y, batch_x_mark, batch_y_mark = batch_data

                outputs, batch_y_true, _, loss_dict, gates = self._process_one_batch(
                    batch_data, self.criterion, phase='test', return_gates=True,
                    curriculum_ratio=1.0, current_prior_weight=0.0
                )

                if gates:
                    detailed_gates['load'].append(gates.get('load', 0))
                    detailed_gates['pv'].append(gates.get('pv', 0))
                    detailed_gates['wind'].append(gates.get('wind', 0))
                detailed_gates['pv_true'].append(batch_y_true[:, :, 1].mean().item())
                detailed_gates['timestamps'].append(i)

                # 只取第一个样本 (Sample 0) 避免平均化模糊
                if i < 5:
                    if gates is not None and 'pv_seq_batch' in gates:
                        vis_data['gate_pv'].append(gates['pv_seq_batch'][0])
                        vis_data['gate_wind'].append(gates['wind_seq_batch'][0])

                        vis_data['irr'].append(gates['irr_seq_batch'][0])
                        vis_data['speed'].append(gates['speed_seq_batch'][0])

                for k in phys_metrics.keys():
                    if k in loss_dict:
                        phys_metrics[k].append(loss_dict[k])

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y_true.detach().cpu().numpy())
                # 收集未来天气数据用于极端场景测试
                weather_data_list.append(batch_weather_future.detach().cpu().numpy())

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
            print(f"[DEBUG] After inverse_transform: preds.min={preds.min():.3f}, preds.max={preds.max():.3f}")
            print(f"[DEBUG] actual_ramp_limits={actual_ramp_limits}")
        else:
            print("[DEBUG] 反归一化被跳过！metric 将在归一化空间计算！")

        metrics_result = metric(preds, trues, ramp_limits=actual_ramp_limits)
        mae, mse, rmse, bvr, rvr = metrics_result
        
        print("\n" + "=" * 60)
        print("  Standard Prediction Metrics (Test Set)")
        print("=" * 60)
        print(f"  MSE          : {mse:.6f}")
        print(f"  MAE          : {mae:.6f}")
        print(f"  RMSE         : {rmse:.6f}")
        print("=" * 60)

        # [FIX] 摒弃错误的 loss_dict 平均值，调用真实的物理合规性计算工具
        compliance = PhysicsComplianceMetrics(
            means=self.train_means,
            stds=self.train_stds,
            ramp_limits=actual_ramp_limits
        )
        real_compliance_metrics = compliance.compute_all_metrics(
            pred=preds, true=trues, time_interval=0.25
        )

        print("\n" + "=" * 60)
        print("  VPP Physics Compliance Report (Test Set) [DENORMALIZED]")
        print("=" * 60)
        
        # 挑选最核心的几个打印展示
        display_keys = {
            'dir_mean': 'DIR (Cos Loss)',
            'kcl_residual_mae': 'NET (KCL MAE)',
            'energy_energy_error_mae': 'ENERGY (Err MAE)',
            'violation_violation_rate': 'BVR (%)'
        }
        for k, label in display_keys.items():
            if k in real_compliance_metrics:
                print(f"  {label:<20} : {real_compliance_metrics[k]:.6f}")

        # 计算综合平均 RVR (%)
        rvr_keys = [k for k in real_compliance_metrics.keys() if 'ramp_violation_rate' in k]
        if rvr_keys:
            avg_rvr = np.mean([real_compliance_metrics[k] for k in rvr_keys])
            print(f"  {'RVR (%)':<20} : {avg_rvr:.6f}")
        
        print("=" * 60 + "\n")

        path = str(self.run_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        folder_path = path + '/'

        # 仅当收集到门控可视化数据时才保存（消融变体 w/o PGCC 和 w/o Physics 不生成）
        if len(vis_data['gate_pv']) > 0:
            np.save(os.path.join(folder_path, 'vis_gate_pv.npy'), np.array(vis_data['gate_pv']))
            np.save(os.path.join(folder_path, 'vis_gate_wind.npy'), np.array(vis_data['gate_wind']))
            np.save(os.path.join(folder_path, 'vis_irr.npy'), np.array(vis_data['irr']))
            np.save(os.path.join(folder_path, 'vis_speed.npy'), np.array(vis_data['speed']))

        np.save(os.path.join(folder_path, 'gate_details.npy'), detailed_gates)
        # 将我们真实计算的合规字典保存
        np.save(os.path.join(folder_path, 'phys_metrics.npy'), real_compliance_metrics)
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array(metrics_result))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        # 极端场景测试
        if extreme_scenario_test:
            print("\n" + "="*60)
            print("  极端场景测试 (Extreme Scenario Testing)")
            print("="*60)

            # 检查是否有天气数据
            if len(weather_data_list) == 0:
                print("警告：未收集到天气数据，无法进行极端场景测试")
            else:
                # 合并天气数据
                weather_data = np.concatenate(weather_data_list, axis=0)

                # 确保天气数据形状与预测数据匹配
                # weather_data: [B, S, 3], predictions: [B, T, 3]
                # 注意：天气序列长度(S)可能与预测长度(T)不同
                # 简化处理：假设S=T，如果不匹配则截断或填充
                B_weather, S, _ = weather_data.shape
                B_pred, T, _ = preds.shape

                if B_weather != B_pred:
                    print(f"警告：天气数据批次大小({B_weather})与预测批次大小({B_pred})不匹配")
                    # 截断到最小批次
                    min_batch = min(B_weather, B_pred)
                    weather_data = weather_data[:min_batch]
                    preds_scenario = preds[:min_batch]
                    trues_scenario = trues[:min_batch]
                else:
                    preds_scenario = preds
                    trues_scenario = trues

                # 导入极端场景测试器
                try:
                    from evaluation.extreme_scenarios import ExtremeScenarioTester

                    # 创建测试器实例
                    tester = ExtremeScenarioTester(dataset=test_data, scaler=test_data.scaler if test_data.scale else None)

                    # 评估所有场景
                    all_metrics = tester.evaluate_all_scenarios(
                        predictions=preds_scenario,
                        targets=trues_scenario,
                        weather_data=weather_data
                    )

                    # 打印详细报告
                    tester.print_scenario_report(all_metrics, title="PhysFormer 极端场景性能报告")

                    # 保存极端场景结果
                    extreme_scenario_path = os.path.join(folder_path, 'extreme_scenarios')
                    os.makedirs(extreme_scenario_path, exist_ok=True)

                    # 保存所有场景指标
                    import json
                    # 转换numpy类型为Python原生类型以便JSON序列化
                    def convert_for_json(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.generic):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [convert_for_json(item) for item in obj]
                        else:
                            return obj

                    serializable_metrics = convert_for_json(all_metrics)
                    with open(os.path.join(extreme_scenario_path, 'scenario_metrics.json'), 'w') as f:
                        json.dump(serializable_metrics, f, indent=2)

                    # 保存原始天气数据以供后续分析
                    np.save(os.path.join(extreme_scenario_path, 'weather_data.npy'), weather_data)

                    print(f"极端场景测试结果已保存至: {extreme_scenario_path}")

                except ImportError as e:
                    print(f"导入极端场景测试模块失败: {e}")
                    print("请确保 evaluation/extreme_scenarios.py 存在")
                except Exception as e:
                    print(f"极端场景测试执行失败: {e}")
                    import traceback
                    traceback.print_exc()

        if return_preds:
            return preds, trues, metrics_result
        else:
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

    def __call__(self, val_loss, model, path, optimizer=None, scheduler=None, scaler=None, epoch=None, global_step=None):
        import numpy as np
        # BUGFIX: 防止 val_loss 为 NaN 时短路比较逻辑强制保存损坏的模型
        if not np.isfinite(val_loss):
            if self.logger:
                self.logger.warning(f'Warning: EarlyStopping encountered invalid val_loss: {val_loss}. Skipping save.')
            elif self.verbose:
                print(f'Warning: EarlyStopping encountered invalid val_loss: {val_loss}. Skipping save.')
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        # 这里的 val_loss 实际上是 NRMSE，越小越好
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, epoch, global_step)
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
            self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, epoch, global_step)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, optimizer=None, scheduler=None, scaler=None, epoch=None, global_step=None):
        if self.logger:
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        elif self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        if optimizer is not None:
            state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict() if scaler else None,
                'epoch': epoch,
                'global_step': global_step,
                'best_score': self.best_score,
                'val_loss_min': val_loss,
                'counter': self.counter
            }
            torch.save(state, path + '/' + 'training_state.pth')
            
        self.val_loss_min = val_loss
