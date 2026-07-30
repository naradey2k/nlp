"""
Vanilla Multi-Head Attention, implemented from scratch on top of torch.nn
primitives (no masking, no fused in_proj -- just the core mechanism).

Your job: fill in the methods marked with `# TODO`. Everything else
(module structure, parameter shapes, naming) is fixed so that the test
suite in `test_attention.py` can load weights from a real
`torch.nn.MultiheadAttention` into your module and compare outputs and
gradients directly.

Read attention/README.md first if this is your first time implementing
attention -- it has the formulas and shape bookkeeping you'll need.
"""

from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention, batch-first.

    Shapes follow torch.nn.MultiheadAttention(..., batch_first=True):
      query: (batch, tgt_len, embed_dim)
      key:   (batch, src_len, embed_dim)
      value: (batch, src_len, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        """
        :param embed_dim: total dimension of the model (d_model)
        :param num_heads: number of attention heads. Must divide embed_dim evenly.
        :param bias: whether the q/k/v/out projections use an additive bias
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate projections (rather than one fused in_proj) so shapes stay
        # obvious. The test file will slice a real nn.MultiheadAttention's
        # fused in_proj_weight/in_proj_bias into these three.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, seq_len, embed_dim)
        :return: (batch, num_heads, seq_len, head_dim)
        """
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, num_heads, seq_len, head_dim)
        :return: (batch, seq_len, embed_dim)
        """
        B, H, T, C = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * C)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param query: (batch, tgt_len, embed_dim)
        :param key: (batch, src_len, embed_dim)
        :param value: (batch, src_len, embed_dim)
        :param attn_mask: optional, broadcastable to (batch, num_heads, tgt_len, src_len)
            (e.g. (tgt_len, src_len) for a causal mask shared across the batch).
            Bool tensor where True means "disallow this (query, key) pair".
        :return: (output, attn_weights)
            output: (batch, tgt_len, embed_dim)
            attn_weights: (batch, tgt_len, src_len), averaged over heads
                (matches nn.MultiheadAttention's default)
        """
        # 1. Project inputs, then split into heads:
        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        # TODO
        B, T, _ = query.size()

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # 2. Scaled dot-product scores: q @ k^T / sqrt(head_dim)
        # -> (batch, num_heads, tgt_len, src_len)
        # TODO
        scores = q @ k.transpose(2, 3) / self.head_dim**0.5

        # 3. Apply attn_mask, if given: set masked-out positions to -inf
        # before softmax, so they get ~0 probability.
        # TODO
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores += attn_mask

        # 4. Softmax over the last dimension (src_len) to get attention
        # weights, shape (batch, num_heads, tgt_len, src_len).
        # TODO
        attn_weights = nn.Softmax(dim=3)(scores)

        # 5. Weighted sum of values: (batch, num_heads, tgt_len, head_dim)
        # TODO
        attn_output = attn_weights @ v

        # 6. Merge heads back to (batch, tgt_len, embed_dim) and apply the
        # output projection.
        # TODO
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)

        output = self.out_proj(attn_output)

        # nn.MultiheadAttention's default is to average attention weights
        # over heads before returning them.
        attn_weights = attn_weights.mean(dim=1)

        return output, attn_weights
