"""Offline weight quantization -- YOUR SOLUTION.

Modify this file to implement your own quantization strategy.
The standard implementation uses round-to-nearest symmetric INT4
with group_size=64. You may change the algorithm or the group_size, as long as:
  1. The function signature stays the same.
  2. The output format is compatible with your CUDA gemm_int4 kernel.
  3. The end-to-end result passes the cosine similarity threshold (>0.98).


The packed format convention:
  - Two signed INT4 values per uint8 byte
  - Low nibble = even element, high nibble = odd element
  - Scales are FP16, one per group
"""

import torch


def quantize_weights(weight: torch.Tensor, group_size: int = 64,
                     clip_percentile: float = 0.9999) -> dict:
    """Quantize a FP16 weight tensor to packed INT4 format.

    Args:
        weight: [N, K] float16 weight tensor.
        group_size: Number of elements per quantization group.
        clip_percentile: Percentile threshold for robust outlier suppression.
            0.9999 clips ~0 values per group-64. Use 1.0 for hard max (reference).

    Returns:
        dict with:
            "weight_packed": [N, K//2] uint8 tensor (packed INT4)
            "weight_scales": [N, K//group_size] float16 tensor (per-group scales)
            "group_size": int
    """
    assert weight.dim() == 2, "weight must be 2D [N, K]"
    N, K = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size % 2 == 0, "group_size must be even"

    num_groups = K // group_size

    # Work in float32 for precision
    w = weight.float().reshape(N, num_groups, group_size)

    # A2: percentile clipping — robust to outliers vs hard max
    # clip_percentile=0.9999 clips ~0 values per group-64 but anchors scale
    # away from single extreme values that would inflate the group scale.
    w_flat = w.abs().reshape(N * num_groups, group_size)
    clip_val = torch.quantile(w_flat, clip_percentile, dim=-1)   # [N*num_groups]
    max_abs = clip_val.reshape(N, num_groups, 1).clamp(min=1e-8)

    # A1: denominator 7.5 uses full INT4 range [-8, +7] (ref used 7.0, wasting -8)
    # Must match the online kernel's denominator (kernel.cu uses 7.5).
    scale = max_abs / 7.5
    rscale = 7.5 / max_abs   # max_abs already clamped > 0

    # Quantize: round to nearest, clamp to [-8, 7]
    q = (w * rscale).round().clamp(-8, 7).to(torch.int8)  # [N, num_groups, group_size]
    q = q.reshape(N, K)

    # Pack two INT4 values per byte: low nibble = even, high nibble = odd
    even = (q[:, 0::2] & 0xF).to(torch.uint8)
    odd  = ((q[:, 1::2] & 0xF) << 4).to(torch.uint8)
    packed = odd | even  # [N, K//2]

    return {
        "weight_packed": packed,
        "weight_scales": scale.squeeze(-1).half(),
        "group_size": group_size,
    }
                       
