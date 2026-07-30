"""
Cross-checks your MultiHeadAttention against torch.nn.MultiheadAttention.

Run directly:
    uv run python attention/test_attention.py
or from inside the attention/ folder:
    python test_attention.py
"""

import torch
from torch import nn

from attention import MultiHeadAttention


def _load_reference_weights(mine: MultiHeadAttention, ref: nn.MultiheadAttention):
    """Copy a real nn.MultiheadAttention's fused in_proj weights into our
    separate q_proj/k_proj/v_proj/out_proj so both modules compute the
    exact same function."""
    with torch.no_grad():
        w_q, w_k, w_v = ref.in_proj_weight.chunk(3, dim=0)
        mine.q_proj.weight.copy_(w_q)
        mine.k_proj.weight.copy_(w_k)
        mine.v_proj.weight.copy_(w_v)

        if ref.in_proj_bias is not None:
            b_q, b_k, b_v = ref.in_proj_bias.chunk(3, dim=0)
            mine.q_proj.bias.copy_(b_q)
            mine.k_proj.bias.copy_(b_k)
            mine.v_proj.bias.copy_(b_v)

        mine.out_proj.weight.copy_(ref.out_proj.weight)
        if ref.out_proj.bias is not None:
            mine.out_proj.bias.copy_(ref.out_proj.bias)


def _make_modules(embed_dim, num_heads, bias, seed):
    torch.manual_seed(seed)
    ref = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)
    mine = MultiHeadAttention(embed_dim, num_heads, bias=bias)
    _load_reference_weights(mine, ref)
    return mine, ref


def _assert_close(name, a, b, atol=1e-5, rtol=1e-4):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"{name} mismatch, max abs diff = {max_diff:.3e}")


def test_self_attention_forward():
    print("test_self_attention_forward ... ", end="")
    for embed_dim, num_heads in [(16, 1), (16, 2), (32, 4), (64, 8)]:
        for bias in (True, False):
            mine, ref = _make_modules(embed_dim, num_heads, bias, seed=0)
            batch, seq_len = 4, 7
            x = torch.randn(batch, seq_len, embed_dim)

            ref_out, ref_w = ref(x, x, x, need_weights=True, average_attn_weights=True)
            my_out, my_w = mine(x, x, x)

            _assert_close(
                f"output (d={embed_dim}, h={num_heads}, bias={bias})", my_out, ref_out
            )
            _assert_close(
                f"attn_weights (d={embed_dim}, h={num_heads}, bias={bias})", my_w, ref_w
            )
    print("OK")


def test_cross_attention_forward():
    print("test_cross_attention_forward ... ", end="")
    embed_dim, num_heads = 32, 4
    mine, ref = _make_modules(embed_dim, num_heads, bias=True, seed=1)
    batch, tgt_len, src_len = 3, 5, 9
    query = torch.randn(batch, tgt_len, embed_dim)
    key = torch.randn(batch, src_len, embed_dim)
    value = torch.randn(batch, src_len, embed_dim)

    ref_out, ref_w = ref(
        query, key, value, need_weights=True, average_attn_weights=True
    )
    my_out, my_w = mine(query, key, value)

    _assert_close("cross-attention output", my_out, ref_out)
    _assert_close("cross-attention attn_weights", my_w, ref_w)
    print("OK")


def test_causal_attn_mask():
    print("test_causal_attn_mask ... ", end="")
    embed_dim, num_heads = 32, 4
    mine, ref = _make_modules(embed_dim, num_heads, bias=True, seed=3)
    batch, seq_len = 2, 6
    x = torch.randn(batch, seq_len, embed_dim)

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    ref_out, ref_w = ref(
        x, x, x, attn_mask=causal_mask, need_weights=True, average_attn_weights=True
    )
    my_out, my_w = mine(x, x, x, attn_mask=causal_mask)

    _assert_close("causal-masked output", my_out, ref_out)
    _assert_close("causal-masked attn_weights", my_w, ref_w)

    # sanity check: no attention leaks into future positions
    assert torch.all(
        my_w[:, torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 1] < 1e-6
    )
    print("OK")


def test_gradients():
    print("test_gradients ... ", end="")
    embed_dim, num_heads = 32, 4
    mine, ref = _make_modules(embed_dim, num_heads, bias=True, seed=5)
    batch, seq_len = 4, 6

    x_ref = torch.randn(batch, seq_len, embed_dim, requires_grad=True)
    x_mine = x_ref.detach().clone().requires_grad_(True)

    ref_out, _ = ref(x_ref, x_ref, x_ref, need_weights=False)
    my_out, _ = mine(x_mine, x_mine, x_mine)

    ref_out.sum().backward()
    my_out.sum().backward()

    _assert_close("grad wrt input", x_mine.grad, x_ref.grad)

    ref_params = dict(ref.named_parameters())
    w_q, w_k, w_v = ref_params["in_proj_weight"].grad.chunk(3, dim=0)
    _assert_close("grad q_proj.weight", mine.q_proj.weight.grad, w_q)
    _assert_close("grad k_proj.weight", mine.k_proj.weight.grad, w_k)
    _assert_close("grad v_proj.weight", mine.v_proj.weight.grad, w_v)
    _assert_close(
        "grad out_proj.weight",
        mine.out_proj.weight.grad,
        ref_params["out_proj.weight"].grad,
    )
    print("OK")


def run_all():
    test_self_attention_forward()
    test_cross_attention_forward()
    test_causal_attn_mask()
    test_gradients()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()
