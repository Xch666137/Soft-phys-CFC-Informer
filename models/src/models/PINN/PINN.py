import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Simple Physics-Informed Neural Network (Baseline)
    本质是一个 MLP，但作为一个对照组，用于验证"架构 vs Loss"的贡献。
    """

    def __init__(self, configs):
        super(PINN, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # 输入维度：Flatten 后的历史序列
        self.input_dim = self.seq_len * self.enc_in
        # 输出维度：Flatten 后的预测序列
        self.output_dim = self.pred_len * self.c_out

        hidden_dim = 512

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),  # 物理信息网络常选用 Tanh 或 Sin 激活函数以保证高阶导数存在
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  # ReLU 有助于稀疏化
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Enc_In]
        batch_size = x_enc.shape[0]

        # Flatten input
        x_flat = x_enc.reshape(batch_size, -1)

        # Forward
        out_flat = self.net(x_flat)

        # Reshape output: [Batch, Pred_Len, C_Out]
        output = out_flat.view(batch_size, self.pred_len, self.c_out)

        return output