"""
ProbSparse注意力机制实现 (完整理论版)
- 核心度量: 恢复了基于 KL 散度的稀疏度测量 (User Original Implementation)
- 未采样处理: 对未选中的 Query 使用 Mean(V) 进行填充，保持 O(L log L)
- 稳定性修复: 包含 Mask 切片、AMP 类型对齐、RoPE 支持
- 初始化: 使用优化的 User Scheme
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F
from .positional import RoPEPositionalEncoding


def np_log_len(L_K):
    return math.log(L_K) if L_K > 0 else 1

class ProbAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5,
                 use_rope=False, rope_base=10000):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEPositionalEncoding(self.d_k, base=rope_base)

        self._init_weights()

    def _init_weights(self):
        """ProbAttention层专门初始化 (User Scheme)"""
        # QKV: gain=0.5
        for module in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None: nn.init.zeros_(module.bias)
        # Output: gain=0.8
        nn.init.xavier_normal_(self.w_o.weight, gain=0.8)
        if self.w_o.bias is not None: nn.init.zeros_(self.w_o.bias)

    def measure_query_sparsity(self, query: Tensor, key: Tensor, sample_size: int) -> Tensor:
        """
        测量查询稀疏度基于KL散度 (User Original Implementation)
        计算 Attention 分布与 Uniform 分布的 KL 散度
        """
        B, H, L_Q, D = query.shape
        _, _, L_K, _ = key.shape

        # 随机采样键向量 [B, H, sample_size, D]
        # 使用 randint 随机选取索引
        sample_indices = torch.randint(0, L_K, (sample_size,), device=query.device)
        # 扩展 key 以便 gather: [B, H, L_K, D] -> [B, H, 1, L_K, D] -> expand
        K_sample = key[:, :, sample_indices, :]

        # 计算 Q * K_sample^T
        # [B, H, L_Q, D] @ [B, H, D, sample] -> [B, H, L_Q, sample]
        QK = torch.matmul(query, K_sample.transpose(-2, -1)) / math.sqrt(D)

        # 计算注意力分布 Q (Logits -> Softmax)
        attention_dist = F.softmax(QK, dim=-1)

        # 计算均匀分布 P
        uniform_dist = torch.ones_like(attention_dist) / sample_size

        # 计算 KL 散度: D_KL(P || Q) 或 D_KL(Q || P)
        # 论文中通常是用近似方法，这里既然你实现了 KL，我们保留它
        # 注意：KLDivLoss 接收 log_target 为 input
        kl_div = F.kl_div(
            torch.log(uniform_dist + 1e-4), # input (log_probs)
            attention_dist + 1e-4,          # target (probs)
            reduction='none'
        )
        # 对 sample 维度求和得到每个 query 的稀疏性得分
        sparsity_scores = kl_div.sum(dim=-1)  # [B, H, L_Q]

        # 注意：KL散度越大，分布越不均匀（越稀疏），该 Query 越重要
        # 但 pytorch F.kl_div 的正负号取决于参数顺序，这里我们取绝对值或者直接作为分数
        # 你的原实现是 sum，我们直接返回
        return sparsity_scores

    def forward(self, query, key, value, mask=None, rope_offset=0):
        B, L_Q, D = query.shape
        _, L_K, _ = key.shape

        # 1. 投影
        Q = self.w_q(query).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        # 2. RoPE
        if self.use_rope:
            Q = self.rope(Q, offset=rope_offset)
            K = self.rope(K, offset=rope_offset)

        # 3. ProbSparse 决策
        u_calc = self.factor * np_log_len(L_K)
        u = int(u_calc) if u_calc < L_Q else L_Q

        if u >= L_Q: # 序列短 -> Full Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                if mask.dim() == 3: mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, -1e4)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            output = torch.matmul(attn, V)

        else: # 序列长 -> ProbSparse Attention
            # A. 使用 KL 散度计算稀疏分数
            sparsity_scores = self.measure_query_sparsity(Q, K, sample_size=u)

            # B. 选取 Top-u 个最重要的 Query
            # top_indices: [B, H, u]
            _, top_indices = sparsity_scores.topk(u, dim=-1)

            # C. 提取 Top-u Query
            # gather_indices: [B, H, u, d_k]
            gather_indices_Q = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
            Q_reduce = torch.gather(Q, 2, gather_indices_Q) # [B, H, u, d_k]

            # D. 计算这些 Query 的 Attention
            scores_top = torch.matmul(Q_reduce, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # E. 关键修复：Mask 切片
            if mask is not None:
                if mask.dim() == 3: mask = mask.unsqueeze(1)
                mask_expanded = mask.expand(B, self.n_heads, L_Q, L_K)
                gather_indices_mask = top_indices.unsqueeze(-1).expand(-1, -1, -1, L_K)
                mask_slice = torch.gather(mask_expanded, 2, gather_indices_mask)
                scores_top = scores_top.masked_fill(mask_slice == 0, -1e4)

            # F. Softmax & Matmul
            attn_top = self.dropout(torch.softmax(scores_top, dim=-1))
            output_top = torch.matmul(attn_top, V) # [B, H, u, d_k]

            # G. 未采样部分处理 (Unsampled Handling)
            # 策略：被选中的位置填入 Attention 结果，未选中的位置填入 V 的均值 (Mean Value)
            # 这是 Informer 保持信息流动的标准做法

            # 1. 创建全 0 画布，显式指定 dtype 防止 AMP 报错
            output = torch.zeros(B, self.n_heads, L_Q, self.d_k, device=Q.device, dtype=output_top.dtype)

            # 2. 计算 V 的均值作为背景值
            # V_mean: [B, H, 1, d_k] -> broadcast to [B, H, L_Q, d_k]
            V_mean = V.mean(dim=-2, keepdim=True)
            output = output + V_mean # 先全部填满均值

            # 3. 将计算出的 Top-u 结果填入对应的位置 (覆盖均值)
            scatter_idx = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
            output = output.scatter(2, scatter_idx, output_top)

        # 合并头
        output = output.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        return self.w_o(output)


class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False,
                 use_rope=False, rope_base=10000, mask_flag=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)  # 使用传入的 dropout

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.use_rope = use_rope
        if use_rope:
            from .positional import RoPEPositionalEncoding  # 确保引入
            self.rope = RoPEPositionalEncoding(self.d_k, base=rope_base)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.w_q(queries).view(B, L, H, -1).permute(0, 2, 1, 3)
        K = self.w_k(keys).view(B, S, H, -1).permute(0, 2, 1, 3)
        V = self.w_v(values).view(B, S, H, -1).permute(0, 2, 1, 3)

        if self.use_rope:
            Q = self.rope(Q)
            K = self.rope(K)

        scale = self.scale or (self.d_k ** -0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if self.mask_flag and attn_mask is not None:
            if attn_mask.dim() == 3: attn_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        A = self.dropout(torch.softmax(scores, dim=-1))
        V_out = torch.matmul(A, V)

        V_out = V_out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        output = self.w_o(V_out)

        if self.output_attention:
            return output, A
        else:
            return output, None