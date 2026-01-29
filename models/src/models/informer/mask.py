"""
Informer 掩码工具 (精简版)
只保留核心的三角因果掩码，用于 Decoder 防止未来信息泄露。
"""

import torch

class TriangularCausalMask():
    """
    三角因果掩码生成器
    生成一个 [B, 1, L, L] 的布尔掩码

    - 上三角 (未来信息) 为 False (0) -> 会被 mask 掉
    - 下三角 (历史信息) 为 True (1)  -> 会被保留
    """
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # 1. 生成上三角矩阵 (diagonal=1 表示对角线往上一格开始为1)
            # triu_mask 中: 1 代表未来(需要遮挡), 0 代表历史(需要保留)
            triu_mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

            # 2. 取反
            # self.mask 中: 0 代表未来(遮挡), 1 代表历史(保留)
            # 这符合 attention.py 中 scores.masked_fill(mask == 0, -1e9) 的逻辑
            self._mask = ~triu_mask

    @property
    def mask(self):
        return self._mask