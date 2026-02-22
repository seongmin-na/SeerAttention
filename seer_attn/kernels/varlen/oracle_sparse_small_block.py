import torch
import triton
import triton.language as tl
import argparse

import math
import time
from einops import rearrange, einsum
from seer_attn.modules.common import repeat_kv
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2]
        for num_stages in [2, 3, 4]
    ],
    key=['BLOCK_H', 'BLOCK_N', 'BLOCK_D'],
)
@triton.jit
def _split_kernel_small_block(
    q_ptr,
    k_cache_ptr,
    cache_seqlens_ptr,
    max_seqlen,
    o_partial_ptr,  ## reuse for qk max
    po_ptr,
    sm_scale,
    gqa_group_size,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_o_b, stride_o_h, stride_o_s,
    stride_po_b, stride_po_h, stride_po_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx_kv = tl.program_id(1)

    head_idx_q = head_idx_kv * gqa_group_size
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)

    cache_seqlens = tl.load(cache_seqlens_ptr + batch_idx)
    cache_leftpad = max_seqlen - cache_seqlens
    leftpad_block_offset = cache_leftpad // BLOCK_N
    num_blocks = (max_seqlen + BLOCK_N - 1) // BLOCK_N - leftpad_block_offset

    q_ptr += batch_idx * stride_q_b + head_idx_q * stride_q_h
    k_cache_ptr += batch_idx * stride_k_b + head_idx_kv * stride_k_h + offs_n[None, :] * stride_k_s + offs_d[:, None] * stride_k_d
    o_partial_ptr += batch_idx * stride_o_b + head_idx_q * stride_o_h
    po_ptr += batch_idx * stride_po_b + head_idx_kv * stride_po_h

    q = tl.load(q_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d, mask=offs_h[:, None] < gqa_group_size)

    # First pass: compute QK scores and store max per block
    for block_idx in range(num_blocks):
        start_n = (leftpad_block_offset + block_idx) * BLOCK_N
        k_ptr = k_cache_ptr + start_n * stride_k_s

        k_mask = (start_n + offs_n[None, :] < max_seqlen) & (start_n + offs_n[None, :] >= cache_leftpad)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)
        # Use element-wise ops instead of tl.dot (tl.dot requires dims >= 16)
        # q: [BLOCK_H, BLOCK_D], k: [BLOCK_D, BLOCK_N]
        # qk[h, n] = sum_d(q[h, d] * k[d, n])
        qk = tl.sum(q[:, :, None].to(tl.float32) * k[None, :, :].to(tl.float32), axis=1)
        qk = qk * sm_scale
        qk = tl.where(start_n + offs_n[None, :] < max_seqlen, qk, float("-inf"))
        qk_max = tl.max(qk, 1)
        o_ptrs = o_partial_ptr + offs_h * stride_o_h + (leftpad_block_offset + block_idx) * stride_o_s
        # store qk_max
        tl.store(o_ptrs, qk_max, mask=offs_h < gqa_group_size)

        m_ij = tl.maximum(m_i, qk_max)
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Second pass: rescale and pool
    for block_idx in range(num_blocks):
        o_ptrs = o_partial_ptr + offs_h * stride_o_h + (leftpad_block_offset + block_idx) * stride_o_s
        po_ptrs = po_ptr + (leftpad_block_offset + block_idx) * stride_po_s
        # rescale qk_max
        local_max = tl.load(o_ptrs, mask=offs_h < gqa_group_size, other=float("-inf"))
        local_max -= m_i
        local_max = tl.exp(local_max) / l_i
        head_pooled = tl.max(local_max, 0)
        tl.store(po_ptrs, head_pooled)


## Alternative version that processes multiple small blocks in one iteration
@triton.jit
def _split_kernel_small_block_v2(
    q_ptr,
    k_cache_ptr,
    cache_seqlens_ptr,
    max_seqlen,
    o_partial_ptr,
    po_ptr,
    sm_scale,
    gqa_group_size,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_o_b, stride_o_h, stride_o_s,
    stride_po_b, stride_po_h, stride_po_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized for BLOCK_N < 16:
    - Uses fewer warps to avoid underutilization
    - Vectorized loads where possible
    - Minimal register pressure
    """
    batch_idx = tl.program_id(0)
    head_idx_kv = tl.program_id(1)

    head_idx_q = head_idx_kv * gqa_group_size
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)

    cache_seqlens = tl.load(cache_seqlens_ptr + batch_idx)
    cache_leftpad = max_seqlen - cache_seqlens
    leftpad_block_offset = cache_leftpad // BLOCK_N
    num_blocks = (max_seqlen + BLOCK_N - 1) // BLOCK_N - leftpad_block_offset

    q_ptr += batch_idx * stride_q_b + head_idx_q * stride_q_h
    k_cache_ptr += batch_idx * stride_k_b + head_idx_kv * stride_k_h
    o_partial_ptr += batch_idx * stride_o_b + head_idx_q * stride_o_h
    po_ptr += batch_idx * stride_po_b + head_idx_kv * stride_po_h

    q = tl.load(q_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d, mask=offs_h[:, None] < gqa_group_size)

    # First pass: compute QK and local max
    for block_idx in range(num_blocks):
        start_n = (leftpad_block_offset + block_idx) * BLOCK_N

        # Load K for this block
        k_ptrs = k_cache_ptr + (start_n + offs_n[None, :]) * stride_k_s + offs_d[:, None] * stride_k_d
        k_mask = (start_n + offs_n[None, :] < max_seqlen) & (start_n + offs_n[None, :] >= cache_leftpad)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute attention scores using element-wise ops (tl.dot requires dims >= 16)
        # q: [BLOCK_H, BLOCK_D], k: [BLOCK_D, BLOCK_N]
        # qk[h, n] = sum_d(q[h, d] * k[d, n])
        qk = tl.sum(q[:, :, None].to(tl.float32) * k[None, :, :].to(tl.float32), axis=1) * sm_scale
        qk = tl.where((start_n + offs_n[None, :] < max_seqlen) & (start_n + offs_n[None, :] >= cache_leftpad), qk, float("-inf"))

        # Compute and store max
        qk_max = tl.max(qk, 1)
        o_ptrs = o_partial_ptr + offs_h * stride_o_h + (leftpad_block_offset + block_idx) * stride_o_s
        tl.store(o_ptrs, qk_max, mask=offs_h < gqa_group_size)

        # Update running statistics
        m_ij = tl.maximum(m_i, qk_max)
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Second pass: normalize and pool across heads
    for block_idx in range(num_blocks):
        o_ptrs = o_partial_ptr + offs_h * stride_o_h + (leftpad_block_offset + block_idx) * stride_o_s
        po_ptrs = po_ptr + (leftpad_block_offset + block_idx) * stride_po_s

        local_max = tl.load(o_ptrs, mask=offs_h < gqa_group_size, other=float("-inf"))
        local_max = tl.exp(local_max - m_i) / l_i
        head_pooled = tl.max(local_max, 0)
        tl.store(po_ptrs, head_pooled)


def oracle_sparse_small_block(
    q,
    k_cache,
    cache_seqlens,
    block_size=8,  # Default to small block size
    sm_scale=None,
    use_v2=False,  # Use alternative kernel version
):
    """
    Oracle sparse attention optimized for small block sizes (< 16).

    Args:
        q: Query tensor (batch, heads, dim) or (batch, 1, heads, dim)
        k_cache: Key cache (batch, max_seqlen, heads_kv, dim)
        cache_seqlens: Actual sequence lengths (batch,)
        block_size: Size of blocks, should be < 16 for this version
        sm_scale: Softmax scaling factor
        use_v2: Whether to use the v2 kernel variant

    Returns:
        po: Pooled attention scores (batch, heads_kv, num_blocks)
    """
    assert block_size < 16, f"This kernel is optimized for block_size < 16, got {block_size}"

    if q.dim() == 4:
        assert q.shape[1] == 1, "q length should be 1"
        q = q.squeeze(1)

    batch, heads, dim = q.shape

    if sm_scale is None:
        sm_scale = 1 / math.sqrt(dim)

    _, max_cache_seqlen, heads_kv, dim_v = k_cache.shape
    group_size = heads // heads_kv

    max_selected_blocks = (max_cache_seqlen + block_size - 1) // block_size

    o_partial = torch.zeros((batch, heads, max_selected_blocks), device=q.device, dtype=torch.float32)
    po = torch.zeros((batch, heads_kv, max_selected_blocks), device=q.device, dtype=q.dtype)

    with torch.cuda.device(q.device.index):
        BLOCK_D = dim
        # For small blocks, use smaller BLOCK_H to reduce register pressure
        BLOCK_H = min(group_size if group_size > 4 else 4, 8)
        grid = (batch, heads_kv)

        kernel = _split_kernel_small_block_v2 if use_v2 else _split_kernel_small_block

        kernel[grid](
            q,
            k_cache,
            cache_seqlens,
            max_cache_seqlen,
            o_partial,
            po,
            sm_scale,
            group_size,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
            po.stride(0), po.stride(1), po.stride(2),
            BLOCK_H=BLOCK_H,
            BLOCK_N=block_size,
            BLOCK_D=BLOCK_D,
        )

    return po


def ref_program(q, k, attention_mask, block_size):
    if q.dim() == 4:
        batch_size, q_len, num_q_heads, head_dim = q.shape
        assert q_len == 1, "q length should be 1"
        q = q.squeeze(1)
    else:
        batch_size, num_q_heads, head_dim = q.shape
        q_len = 1
    _, kv_len, num_kv_heads, _ = k.shape
    num_gqa_groups = num_q_heads // num_kv_heads

    q = q.contiguous()
    k = k.transpose(1, 2).contiguous()

    # Repeat K heads for GQA compatibility
    if num_gqa_groups > 1:
        k = repeat_kv(k, num_gqa_groups)

    attn_weights = torch.einsum('bhd, bhdj -> bhj', q, k.transpose(-1, -2)) * (head_dim**-0.5)

    attention_mask = attention_mask.unsqueeze(1)

    attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))

    attn_weights_qhead = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_weights = F.max_pool2d(attn_weights_qhead, kernel_size=(num_gqa_groups, block_size), stride=(num_gqa_groups, block_size), ceil_mode=True)

    return attn_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='heads_kv')
    parser.add_argument('--max_cache_seqlen', type=int, default=8192, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--block_size', type=int, default=8, help='block_size (should be < 16)')
    parser.add_argument('--use_v2', action='store_true', help='use v2 kernel')
    args = parser.parse_args()

    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v
    block_size = args.block_size

    assert block_size < 16, "This script is for testing small block sizes (< 16)"

    dtype = torch.float16
    num_blocks = (max_cache_seqlen + block_size - 1) // block_size

    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda')
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device='cuda')

    seq_range = torch.arange(max_cache_seqlen, device='cuda')
    attention_mask = seq_range[None, :].ge(max_cache_seqlen - cache_seqlens[:, None])

    print(f"Testing with block_size={block_size}, batch={batch}, heads={heads}, heads_kv={heads_kv}, seqlen={max_cache_seqlen}")
    print("attention_mask:", attention_mask.shape, "cache_seqlens:", cache_seqlens)

    ref = ref_program(Q, K, attention_mask, block_size)

    triton_out = oracle_sparse_small_block(
        Q,
        K,
        cache_seqlens,
        block_size,
        use_v2=args.use_v2,
    )

    print("ref shape:", ref.shape)
    print("triton_out shape:", triton_out.shape)
    print("max diff:", torch.max(torch.abs(ref - triton_out)))

    mismatch_indices = torch.where(torch.abs(ref - triton_out) > 0.1)
    print(f"Number of mismatches (>0.1): {len(mismatch_indices[0])}")

    if torch.allclose(ref, triton_out, atol=1e-2, rtol=1e-2):
        print("✓ Pass test reference implementation.")
    else:
        print("✗ Failed test reference implementation.")
        print("Mismatch locations:", mismatch_indices)

    # Benchmark
    print("\nBenchmarking...")
    for _ in range(10):
        oracle_sparse_small_block(Q, K, cache_seqlens, block_size, use_v2=args.use_v2)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        oracle_sparse_small_block(Q, K, cache_seqlens, block_size, use_v2=args.use_v2)
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) / 100
    print(f"Triton kernel ({'v2' if args.use_v2 else 'v1'}) time: {triton_time*1000:.3f} ms")

    for _ in range(10):
        ref_program(Q, K, attention_mask, block_size)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref_program(Q, K, attention_mask, block_size)
    torch.cuda.synchronize()
    end = time.time()
    ref_time = (end - start) / 100
    print(f"Reference kernel time: {ref_time*1000:.3f} ms")
    print(f"Speedup: {ref_time / triton_time:.2f}x")
