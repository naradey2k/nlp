# Multi-Head Attention — implementation exercise

Goal: implement `MultiHeadAttention` in `attention.py` so that it is
numerically identical to `torch.nn.MultiheadAttention` — same outputs,
same attention weights, same gradients. `test_attention.py` checks this
by copying a real `nn.MultiheadAttention`'s weights into your module and
diffing the two.

## 1. The math

Given queries `Q`, keys `K`, values `V` (already linearly projected),
single-head scaled dot-product attention is:

```
Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V
```

- `Q`: (tgt_len, d_k), `K`, `V`: (src_len, d_k)
- `Q Kᵀ`: (tgt_len, src_len) — a similarity score between every query and
  every key
- dividing by `sqrt(d_k)` keeps the dot products (and thus the softmax)
  from saturating as `d_k` grows — without it, gradients vanish
- `softmax` (over the src_len axis) turns scores into a probability
  distribution per query row
- multiplying by `V` produces a weighted average of value vectors, weighted
  by how relevant each key was to the query

**Multi-head** attention runs `num_heads` independent copies of this in
parallel on slices of the embedding, then concatenates and re-projects:

```
head_i = Attention(Q Wq_i, K Wk_i, V Wv_i)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) Wo
```

In practice you don't loop over heads in Python — you reshape the
projected `(batch, seq_len, embed_dim)` tensor into
`(batch, num_heads, seq_len, head_dim)` and let batched matmul handle all
heads at once. That reshape/un-reshape is exactly `_split_heads` /
`_merge_heads` in the stub.

## 2. Shape bookkeeping (the part people actually get wrong)

- Input: `(batch, seq_len, embed_dim)`, `embed_dim = num_heads * head_dim`
- After projection + split: `(batch, num_heads, seq_len, head_dim)`
- Scores `Q @ Kᵀ`: matmul over the last two dims, batched over
  `(batch, num_heads)` → `(batch, num_heads, tgt_len, src_len)`
- Attention output `weights @ V`: `(batch, num_heads, tgt_len, head_dim)`
- Merge heads back to `(batch, tgt_len, embed_dim)`, then apply `out_proj`

The one gotcha: after `.transpose(1, 2)` the tensor is non-contiguous, so
`.view(...)` will raise. Use `.contiguous().view(...)` or `.reshape(...)`.

Note `Q Kᵀ` doesn't require `tgt_len == src_len` — it only requires `Q`
and `K` to share `head_dim`. So query and key/value are allowed to come
from sequences of different lengths (e.g. a decoder's target tokens
attending to an encoder's source tokens). When query, key, and value are
literally the same tensor (self-attention), `tgt_len == src_len`
trivially, but the shapes stay separate because the mechanism itself
doesn't assume it.

## 3. Masking (`attn_mask`)

Used for causal/autoregressive masking: a query at position `i` shouldn't
be able to attend to a key at position `j > i` (no peeking at the
future). `attn_mask` is a bool tensor, `True` meaning "disallow this
(query, key) pair" — set the corresponding score to `-inf` before the
softmax, so `exp(-inf) = 0` and that position gets zero attention
weight.

## 4. Steps to implement

Work through `attention.py` top to bottom:

1. `_split_heads`: `(batch, seq_len, embed_dim) → (batch, num_heads, seq_len, head_dim)`
2. `_merge_heads`: the inverse
3. In `forward`:
   - project `query`/`key`/`value` through `q_proj`/`k_proj`/`v_proj`, then split heads
   - compute scaled scores
   - apply `attn_mask` if given
   - softmax → `attn_weights`
   - `attn_weights @ v` → merge heads → `out_proj`
   - average `attn_weights` over heads (matches `nn.MultiheadAttention`'s default)

Don't touch `test_attention.py` — it's your oracle.

## 5. Running the tests

```
uv run python attention/test_attention.py
```

Tests run roughly in order of difficulty:

1. `test_self_attention_forward` — plain self-attention, no mask
2. `test_cross_attention_forward` — query/key/value have different lengths
3. `test_causal_attn_mask` — bool `attn_mask`
4. `test_gradients` — backprop through your module must match `nn.MultiheadAttention`'s
   gradients on the input and on `q_proj`/`k_proj`/`v_proj`/`out_proj` weights

If an early test fails, fix it before moving on — later tests build on
the same code path. When all four print `OK`, you're numerically
equivalent to PyTorch's implementation.
