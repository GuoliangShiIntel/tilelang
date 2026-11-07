"""Quantised-weight DPU op of the Intel NPU dialect.

`gemm_i4(a, wq, s, out, group)`: out[M, N] = a[M, K] . dequant(wq[N, K])^T, with `wq` packed signed int4
(two per byte, low nibble first, uint8 [N, K/2]) and fp16 scales `s` [N, K/group]. The DPU reads the int4
weights in place (its mixed fp16 x int4 mode, one workload over all (row, group) channels) and the SHAVE
sums the group partials with the scales: the decode (M = 1) path, where dequantising first would cost the
SHAVE ten times the weight DMA. Requires K % group == 0, group % 16 == 0, N % 16 == 0 and N x (K / group)
<= the DPU's channel limit (1024 in the runtime).
"""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import Buffer

__all__ = ["gemm_i4"]


def _addr(buf: Buffer, idx=None):
    return tirx.address_of(tirx.BufferLoad(buf, idx if idx is not None else [0] * len(buf.shape)))


def gemm_i4(a: Buffer, wq: Buffer, s: Buffer, out: Buffer, group: int = 128, row0=0, rows: int | None = None):
    """`row0`/`rows`: the block of weight rows (and output columns) this workload covers, so one streamed chunk
    of many rows is consumed by several workloads under the channel limit (rows x K / group <= 1024)."""
    M, K = (int(v) for v in a.shape)
    N = int(wq.shape[0]) if rows is None else int(rows)
    if int(wq.shape[1]) * 2 != K:
        raise ValueError(f"intel_npu: gemm_i4: wq {list(wq.shape)} is not a packed int4 [N, K/2] for K={K}")
    if int(s.shape[1]) != K // group or int(s.shape[0]) != int(wq.shape[0]):
        raise ValueError(f"intel_npu: gemm_i4: scales {list(s.shape)} are not [{wq.shape[0]}, K/group = {K // group}]")
    if int(out.shape[0]) != M or int(out.shape[1]) != int(wq.shape[0]):
        raise ValueError(f"intel_npu: gemm_i4: out {list(out.shape)} is not [M, N] = [{M}, {wq.shape[0]}]")
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_dpu_matmul_i4g"),
                            _addr(out, [0, row0]), _addr(a), _addr(wq, [row0, 0]), _addr(s, [row0, 0]), M, N, K, group)
