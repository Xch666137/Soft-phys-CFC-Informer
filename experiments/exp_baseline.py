import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from experiments.exp_PhysFormer import Exp_PhysFormer, EarlyStopping
from models.src.models.informer import Informer
from models.src.models.Autoformer import Autoformer
from models.src.models.LSTM import LSTM
from models.src.models.GRU import GRU
from models.src.models.PINN import PINN
from models.src.utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Baselines(Exp_PhysFormer):
    """
    针对 VPP 高渗透率数据定制的 Baseline 实验控制器
    修改：统一了 Log 系统和 tqdm 进度条
    """

    def __init__(self, args):
        # 调用父类初始化，这里会执行 _init_logger 创建 self.logger
        super().__init__(args)

        # 强制修正参数以适配 VPP 数据集结构
        self.args.enc_in = 6  # Encoder 输入维度 (含气象)
        self.args.dec_in = 6  # Decoder 输入维度 (含气象)
        self.args.c_out = 3  # 输出维度 (只预测 Load, PV, Wind)

    def _build_model(self):
        model_dict = {
            'Informer': Informer,
            'Autoformer': Autoformer,
            'LSTM': LSTM,
            'GRU': GRU,
            'PINN': PINN
        }

        if self.args.model not in model_dict:
            raise ValueError(f"Model {self.args.model} not implemented")

        Model = model_dict[self.args.model]

        # 针对 Transformer 类的特殊配置
        if self.args.model in ['Autoformer', 'Informer']:
            if not hasattr(self.args, 'moving_avg'): self.args.moving_avg = 5
            if not hasattr(self.args, 'output_attention'): self.args.output_attention = False

        # 注意：这里使用 self.args 传递配置，这就要求 core.py 等文件必须适配 configs 输入
        model = Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model.to(self.device)

    def _select_optimizer(self):
        wd = self.args.weight_decay if self.args.weight_decay > 0 else 1e-4
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=wd
        )
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                    self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self.args.model in ['Informer', 'Autoformer']:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 预测对齐
                outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                batch_y = batch_y[:, -self.args.pred_len:, :self.args.c_out]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        criterion = self._select_criterion()
        optimizer = self._select_optimizer()

        # 2. 学习率调度器: 余弦退火 + 热重启
        # 这种调度器允许模型在约束加入时有能力调整权重
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.args.train_epochs,
            eta_min=1e-6  # 最小保底 LR
        )

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

        use_amp = getattr(self.args, 'use_amp', True) and self.args.use_gpu
        scaler = GradScaler(enabled=use_amp)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            with tqdm(total=len(train_loader),
                      desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                      mininterval=0.3,
                      leave=False,
                      ncols=100) as pbar:

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    optimizer.zero_grad()
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                        self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                    with autocast(enabled=use_amp):
                        if self.args.model in ['Informer', 'Autoformer']:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention: outputs = outputs[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # 裁剪输出以匹配 Target (Load, PV, Wind)
                        outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                        batch_y = batch_y[:, -self.args.pred_len:, :self.args.c_out]

                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    pbar.update(1)

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            scheduler.step()

            epoch_cost_time = time.time() - epoch_time

            # [修改] 使用 self.logger 记录日志，格式对齐 PhysFormer
            self.logger.info(
                f"Epoch: {epoch + 1} | Cost: {epoch_cost_time:.2f}s | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        preds = []
        trues = []
        scaler = test_data.scaler

        # [新增] Test 阶段也可以加个简单的 tqdm，或者直接保持静默
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                    self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self.args.model in ['Informer', 'Autoformer']:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention: outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :self.args.c_out]
                batch_y = batch_y[:, -self.args.pred_len:, :self.args.c_out]

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        self.logger.info(">>> Performing Inverse Scaling for VPP Data <<<")

        N, L, C = preds.shape
        preds_flat = preds.reshape(-1, C)
        trues_flat = trues.reshape(-1, C)

        dummy_zeros = np.zeros((preds_flat.shape[0], 3))
        preds_full = np.concatenate([preds_flat, dummy_zeros], axis=1)
        trues_full = np.concatenate([trues_flat, dummy_zeros], axis=1)

        if scaler is not None:
            preds_phys = scaler.inverse_transform(preds_full)[:, :3]
            trues_phys = scaler.inverse_transform(trues_full)[:, :3]
        else:
            preds_phys = preds_flat
            trues_phys = trues_flat

        # 物理修正
        # preds_phys[preds_phys < 0] = 0.0

        preds_phys = preds_phys.reshape(N, L, C)
        trues_phys = trues_phys.reshape(N, L, C)

        mae, mse, rmse, mape, mspe, bvr, rvr = metric(preds_phys, trues_phys)

        # [修改] 使用 self.logger 输出最终指标
        self.logger.info(f'MSE:{mse:.6f}, MAE:{mae:.6f}')
        self.logger.info(f"Load MSE: {np.mean((preds_phys[:, :, 0] - trues_phys[:, :, 0]) ** 2):.6f}")
        self.logger.info(f"PV   MSE: {np.mean((preds_phys[:, :, 1] - trues_phys[:, :, 1]) ** 2):.6f}")
        self.logger.info(f"Wind MSE: {np.mean((preds_phys[:, :, 2] - trues_phys[:, :, 2]) ** 2):.6f}")
        self.logger.info(f'Physics Violation -> BVR:{bvr:.2f}%, RVR:{rvr:.2f}%')

        # 保存结果
        folder_path = './exp_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 修改 exp_baseline.py 最后几行
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, bvr, rvr]))
        np.save(folder_path + 'pred.npy', preds_phys)
        np.save(folder_path + 'true.npy', trues_phys)

        return