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
from models.src.utils.losses import VPPDomainLoss
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
        # 逻辑：取训练集一阶差分的 99.5% 分位数
        try:
            self.logger.info(">>> [Auto-Stat] Calculating Ramp Limits from Training Data...")

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
            real_ramp_limits = np.percentile(diff, 99.5, axis=0)

            # 稍微给一点点裕度 (1.0~1.1倍)，防止训练太敏感
            # 比如：如果历史最大爬坡是 1.0，我们允许模型学到 1.05，但不允许太离谱
            real_ramp_limits = real_ramp_limits * 1.05

            self.logger.info(f">>> [Auto-Stat] Calculated Ramp Limits: {real_ramp_limits}")

        except Exception as e:
            self.logger.warning(f"Failed to calc ramp limits: {e}, using defaults.")
            real_ramp_limits = None

        # 4. 将真实统计量传入 Loss
        criterion = VPPDomainLoss(
            device=self.device,
            means=real_means,
            stds=real_stds,
            ramp_limits=real_ramp_limits
        )
        return criterion

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-4
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=wd
        )
        return model_optim

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, self.args.checkpoint_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # 1. 优化器
        model_optim = self._select_optimizer()

        # 2. 学习率调度器: 余弦退火 + 热重启
        # 这种调度器允许模型在约束加入时有能力调整权重
        scheduler = CosineAnnealingLR(
            model_optim,
            T_max=self.args.train_epochs,
            eta_min=1e-6                    # 最小保底 LR
        )

        # 3. 初始化 Loss (初始物理权重为 0)
        criterion = self._select_criterion()

        # 定义目标权重 (超参数)
        TARGET_DERIV = self.args.w_deriv
        TARGET_ENERGY = self.args.w_energy
        TARGET_BOUND = self.args.w_bound
        TARGET_RAMP = self.args.w_ramp
        TARGET_INERTIA = self.args.w_inertia

        # --- 阶段锚点 ---
        EPOCH_WARMUP = self.args.p_epoch_warmup     # Phase 1 End (Default: 10)
        EPOCH_LOCK = self.args.p_epoch_lock      # Phase 2 End (Default: 30)

        # 4. 早停机制 (监控 物理MAE)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

        for epoch in range(self.args.train_epochs):

            # ============================================================
            # 参数化分阶段策略 (Three-Phase Strategy)
            # ============================================================

            # 初始化当前权重
            cur_deriv = 0.0
            cur_energy = 0.0
            cur_bound = 0.0
            cur_ramp = 0.0
            cur_inertia = 0.0
            phase = ""

            # --- Phase 1: 自由重塑期 (0 ~ WARMUP) ---
            # 策略：高导数权重，低物理约束。让模型先学会"画波形"。
            if epoch < EPOCH_WARMUP:
                phase = "Phase 1: Shape Learning (Derivative Focus)"
                progress = epoch / EPOCH_WARMUP

                # 导数权重：快速上升，强迫模型拟合趋势
                # (从 0.5 开始，快速升到 Target)
                cur_deriv = 0.5 + (TARGET_DERIV - 0.5) * progress

                # 能量：给一点点，防止飘太远
                cur_energy = 0.1 * progress

                # 初期不加惯性约束，允许 CfC 大幅调整时间常数
                cur_inertia = 0.0

                # 边界/爬坡：先不管，避免阻碍波形学习
                cur_bound = 0.0
                cur_ramp = 0.0

            # --- Phase 2: 物理注入期 (WARMUP ~ LOCK) ---
            # 策略：保持波形约束，线性注入硬约束。
            elif epoch < EPOCH_LOCK:
                phase = "Phase 2: Physics Injection (Annealing)"
                # 计算当前阶段的进度 (0.0 -> 1.0)
                progress = (epoch - EPOCH_WARMUP) / (EPOCH_LOCK - EPOCH_WARMUP)

                # 导数：保持高位，锁定波形
                cur_deriv = TARGET_DERIV

                # 能量：线性增长到目标 (0.1 -> Target)
                cur_energy = 0.1 + (TARGET_ENERGY - 0.1) * progress

                # 边界：线性增长 (0.0 -> Target)
                cur_bound = TARGET_BOUND * progress

                # 爬坡：线性增长 (0.0 -> Target)，注意 Target 设小一点(0.1)作为熔断
                cur_ramp = TARGET_RAMP * progress

                # 线性增加惯性约束，平滑动力学
                cur_inertia = TARGET_INERTIA * progress

            # --- Phase 3: 物理锁定期 (LOCK ~ END) ---
            # 策略：全约束生效，模型微调。
            else:
                phase = "Phase 3: Physical Locking & Fine-tuning"
                cur_deriv = TARGET_DERIV
                cur_energy = TARGET_ENERGY
                cur_bound = TARGET_BOUND
                cur_ramp = TARGET_RAMP
                cur_inertia = TARGET_INERTIA

            # 更新 Loss 参数
            criterion.alpha_bound = cur_bound
            criterion.alpha_ramp = cur_ramp
            criterion.alpha_energy = cur_energy
            criterion.alpha_deriv = cur_deriv           # [重要] 确保 criterion 有这个属性

            # 日志打印
            current_lr = model_optim.param_groups[0]['lr']
            self.logger.info(f"\nEpoch {epoch + 1} [{phase}] | LR: {current_lr:.6f}")
            self.logger.info(f"Weights -> Deriv:{cur_deriv:.2f} | Energy:{cur_energy:.2f} | "
                             f"Bound:{cur_bound:.2f} | Ramp:{cur_ramp:.2f} | Inertia:{cur_inertia:.5f}")
            self.model.train()
            train_loss_log = []
            phys_loss_log = {'bound': [], 'ramp': [], 'energy': []}
            epoch_time = time.time()

            with tqdm(total=len(train_loader),
                      desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                      mininterval=0.3,
                      leave=False,
                      ncols=100) as pbar:

                for i, (batch_stat, batch_phys, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    model_optim.zero_grad()

                    # 1. 数据转移
                    batch_stat = batch_stat.float().to(self.device)
                    batch_phys = batch_phys.float().to(self.device)  # 新增
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # 2. 构造 Decoder Input (标准 Informer 范式)
                    # dec_inp = [Start Token (真实值), Place Holder (全0)]
                    # 仅取前 c_out (3) 列作为 Decoder 的输入
                    batch_y_sliced = batch_y[:, :, :self.args.c_out]

                    dec_inp = torch.zeros_like(batch_y_sliced[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y_sliced[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # Forward
                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                        # Forward
                        outputs= self.model(batch_stat, batch_phys, batch_x_mark, dec_inp, batch_y_mark)
                        batch_y_true = batch_y[:, -self.args.pred_len:, :self.args.c_out]

                        # 1. 计算主 Loss (包含 VPP Domain Constraints)
                        loss_main, loss_dict = criterion(outputs, batch_y_true)

                        # 2. 计算惯量正则化 Loss (针对 CFC 参数)
                        loss_inertia = calculate_inertia_loss(self.model, lambda_inertia=cur_inertia)

                        # 3. 总 Loss
                        total_loss = loss_main + loss_inertia

                    # Backward
                    scaler.scale(total_loss).backward()

                    # 梯度裁剪 (关键！防止物理约束导致的梯度爆炸)
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    scaler.step(model_optim)
                    scaler.update()

                    train_loss_log.append(total_loss.item())
                    phys_loss_log['bound'].append(loss_dict['bound'])
                    phys_loss_log['ramp'].append(loss_dict['ramp'])
                    phys_loss_log['energy'].append(loss_dict['energy'])

                    pbar.update(1)

            # --- C. 验证与早停 ---
            # 计算并输出单轮训练耗时
            epoch_cost_time = time.time() - epoch_time

            # 这里的 vali_mae 是反归一化后的真实物理误差 (MW)
            vali_loss, vali_mae = self.vali(val_data, val_loader, criterion)

            # 调度器步进 (WarmRestarts 是按 epoch 更新的)
            scheduler.step()

            self.logger.info(f"Train Loss: {np.mean(train_loss_log):.5f}, Cost time: {epoch_cost_time:.2f}s")
            self.logger.info(
                f"Phys Violations -> Bound: {np.mean(phys_loss_log['bound']):.5f} "
                f"| Ramp: {np.mean(phys_loss_log['ramp']):.5f} |"
                f"| Energy: {np.mean(phys_loss_log['energy']):.5f} ")
            self.logger.info(f"Vali Physical MAE: {vali_mae:.5f}")

            # ============================================================
            #   早停机制的“豁免”与“重置”策略
            # ============================================================

            # 1. 在物理注入期 (Phase 2): 开启“上帝模式”，禁止早停
            if epoch < EPOCH_LOCK:
                # 仍然调用 early_stopping 以便保存那些“偶然变好”的模型权重
                early_stopping(vali_mae, self.model, path)

                # 【核心操作】强制重置计数器
                if early_stopping.counter > 0:
                    self.logger.info(
                        f"  [Immunity] In Injection Phase. Resetting EarlyStopping counter ({early_stopping.counter}/{self.args.patience}).")
                    early_stopping.counter = 0
                    early_stopping.early_stop = False

            # 2. 在物理锁定期的第一刻 (Phase 3 Start): 重置最佳成绩
            elif epoch == EPOCH_LOCK:
                self.logger.info("  [Reset] Constraints Locked. Resetting EarlyStopping Baseline.")
                # 我们要忘掉 Phase 1 那个"不守规矩"的低 MAE，以现在的表现作为新基准
                # 重新初始化 EarlyStopping 对象
                early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)
                # 立即记录当前的 MAE 作为新的 Best Score
                early_stopping(vali_mae, self.model, path)

            # 3. 在物理锁定期 (Phase 3): 恢复正常执法
            else:
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
        total_loss = []

        # 用于记录各项物理违规的平均值
        total_metrics = {'mae': [], 'bound': [], 'ramp': [], 'energy': [], 'deriv': []}

        with torch.no_grad():
            for i, (batch_stat, batch_phys, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_stat = batch_stat.float().to(self.device)
                batch_phys = batch_phys.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder Input
                batch_y_sliced = batch_y[:, :, :self.args.c_out]
                dec_inp = torch.zeros_like(batch_y_sliced[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_sliced[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    # Forward
                    outputs = self.model(batch_stat, batch_phys, batch_x_mark, dec_inp, batch_y_mark)
                    batch_y_true = batch_y[:, -self.args.pred_len:, :self.args.c_out]

                    # Loss 计算 (注意这里解包 tuple)
                    loss, loss_dict = criterion(outputs, batch_y_true)

                total_loss.append(loss.item())

                # 累积各项指标
                for k, v in loss_dict.items():
                    total_metrics[k].append(v)

        avg_loss = np.average(total_loss)
        avg_mae = np.average(total_metrics['mae'])

        # 打印验证集详细物理指标 (可选，方便调试)
        # self.logger.info(f"Vali Breakdown -> Bound: {np.mean(total_metrics['bound']):.4f} "
        #                  f"| Ramp: {np.mean(total_metrics['ramp']):.4f} "
        #                  f"| Energy Error: {np.mean(total_metrics['energy']):.4f} ")

        self.model.train()

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
        phys_metrics = {'bound': [], 'ramp': [], 'energy': [], 'deriv': []}
        # 临时创建一个 criterion 用于计算指标 (权重设为0即可，只为了复用计算逻辑)
        criterion = VPPDomainLoss(device=self.device)

        with torch.no_grad():
            for i, (batch_stat, batch_phys, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_stat = batch_stat.float().to(self.device)
                batch_phys = batch_phys.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # --- 构造 Decoder Input ---
                batch_y_sliced = batch_y[:, :, :self.args.c_out]
                dec_inp = torch.zeros_like(batch_y_sliced[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_sliced[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_stat, batch_phys, batch_x_mark, dec_inp, batch_y_mark)
                    batch_y_true = batch_y[:, -self.args.pred_len:, :self.args.c_out]

                    # 计算物理指标
                    _, loss_dict = criterion(outputs, batch_y_true)

                # 记录指标
                for k in ['bound', 'ramp', 'energy', 'deriv']:
                    phys_metrics[k].append(loss_dict[k])

                # [Batch, Pred_Len, 3]
                pred = outputs.detach().cpu().numpy()
                true = batch_y_true.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        # 拼接所有 Batch
        preds = np.concatenate(preds, axis=0)  # [N, P, 3]
        trues = np.concatenate(trues, axis=0)  # [N, P, 3]

        # --- [修正点 2] 稳健的反归一化逻辑 ---
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
        print("\n" + "=" * 40)
        print("  Physics Compliance Report (Test Set)")
        print("=" * 40)
        print(f"  Avg Bound Violation : {np.mean(phys_metrics['bound']):.6f}")
        print(f"  Avg Ramp Violation  : {np.mean(phys_metrics['ramp']):.6f}")
        print(f"  Avg Energy Error    : {np.mean(phys_metrics['energy']):.6f}")
        print(f"  Avg Deriv Error     : {np.mean(phys_metrics['deriv']):.6f}")
        print("=" * 40 + "\n")

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