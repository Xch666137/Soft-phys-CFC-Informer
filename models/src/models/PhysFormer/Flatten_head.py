import torch
import torch.nn as nn


class FlattenHead(nn.Module):
    """
    Optimized Flatten Head (Channel-Independent)

    原版全连接: [B, S*D] -> [B, P*D] (参数量爆炸: S*P*D^2)
    优化版投影: [B, D, S] -> [B, D, P] (参数量极小: S*P)

    Args:
        seq_len: Input sequence length
        d_model: Model dimension (Not used in weight matrix size anymore)
        pred_len: Prediction sequence length
        dropout: Dropout rate
    """

    def __init__(self, seq_len, d_model, pred_len, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pred_len = pred_len
        self.dropout = nn.Dropout(dropout)

        # 线性层只负责时间步的映射：Seq_len -> Pred_len
        # 参数量: 672 * 96 = 64,512 (只有之前的几十万分之一)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, D_Model]

        # 1. 转置: [B, S, D] -> [B, D, S]
        x = x.permute(0, 2, 1)

        # 2. 投影时间维: [B, D, S] -> [B, D, P]
        x = self.linear(x)
        x = self.dropout(x)

        # 3. 转回: [B, D, P] -> [B, P, D]
        x = x.permute(0, 2, 1)

        return x