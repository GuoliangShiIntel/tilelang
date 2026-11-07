"""Quantised-weight DPU op of the Intel NPU dialect.

`gemm_i4(a, wq, s, out, group)`: out[M, N] = a[M, K] . dequant(wq[N, K])^T, with `wq` packed signed int4
(two per byte, low nibble first, uint8 [N, K/2]) and fp16 scales `s`, either row-major [N, K/group] (the
SHAVE transposes them while the DPU runs) or group-major [K/group, N] (read in place: the packer's layout
for the decode GEMVs, where the transpose of a 256-row chunk with 4 groups cost 18 us per workload -- M3).
The DPU reads the int4 weights in place (its mixed fp16 x int4 mode, one workload over all (row, group)
channels) and the SHAVE sums the group partials with the scales: the decode (M = 1) path, where
dequantising first would cost the SHAVE ten times the weight DMA. Requires K % group == 0, group % 16 == 0,
N % 16 == 0 and N x (K / group) <= the DPU's channel limit (1024 in the runtime).
"""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import Buffer

__all__ = ["gemm_i4", "gemm_i8"]


def _addr(buf: Buffer, idx=None):
    return tirx.address_of(tirx.BufferLoad(buf, idx if idx is not None else [0] * len(buf.shape)))


def _gemm_q(op_name: str, bytes_per_k: float, a: Buffer, wq: Buffer, s: Buffer | None, out: Buffer, group: int, row0, rows,
            scales_major: str | None, scales_row: int | None):
    M, K = (int(v) for v in a.shape)
    ng = K // group
    row_bytes = int(K * bytes_per_k)
    if int(wq.shape[1]) != row_bytes:
        raise ValueError(f"intel_npu: {op_name}: wq {list(wq.shape)} is not [N, {row_bytes}] for K={K}")
    if scales_row is not None:
        # the chunk block: N weight rows, then the group-major fp16 scales [ng, N] in the rows that follow
        if s is not None or scales_major not in (None, "group"):
            raise ValueError(f"intel_npu: {op_name}: scales_row names the scales inside wq (group-major); do not pass s")
        N_all = int(scales_row)
        if (ng * N_all * 2) % row_bytes != 0 or int(wq.shape[0]) != N_all + ng * N_all * 2 // row_bytes:
            raise ValueError(f"intel_npu: {op_name}: wq {list(wq.shape)} is not [{N_all} weight rows + {ng} x {N_all} fp16 "
                             f"scales packed in whole rows of {row_bytes} B]")
        N = N_all if rows is None else int(rows)
    else:
        N_all = int(wq.shape[0])
        N = N_all if rows is None else int(rows)
        s_shape = [int(v) for v in s.shape]
        row_major = s_shape == [N_all, ng]
        group_major = s_shape == [ng, N_all]
        if scales_major is None:
            if row_major and group_major:
                raise ValueError(f"intel_npu: {op_name}: scales {s_shape} could be row- or group-major (N == K/group); "
                                 f"say scales_major='row' or 'group'")
            if not (row_major or group_major):
                raise ValueError(f"intel_npu: {op_name}: scales {s_shape} are neither [N, K/group] = [{N_all}, {ng}] "
                                 f"nor [K/group, N]")
            scales_major = "row" if row_major else "group"
        elif scales_major not in ("row", "group") or not (row_major if scales_major == "row" else group_major):
            raise ValueError(f"intel_npu: {op_name}: scales {s_shape} are not {scales_major}-major for N={N_all}, K/group={ng}")
    if int(out.shape[0]) != M or int(out.shape[1]) != N_all:
        raise ValueError(f"intel_npu: {op_name}: out {list(out.shape)} is not [M, N] = [{M}, {N_all}]")
    if scales_row is not None:
        off = row0 * 2   # bytes into the scales block: column row0 of its first group row
        s_addr, s_stride = _addr(wq, [scales_row + off // row_bytes, off % row_bytes]), N_all
    elif scales_major == "row":
        s_addr, s_stride = _addr(s, [row0, 0]), 0
    else:
        s_addr, s_stride = _addr(s, [0, row0]), N_all
    return tirx.call_intrin("handle", tirx.op.Op.get(f"tl.intel_npu_dpu_matmul_{op_name}"),
                            _addr(out, [0, row0]), _addr(a), _addr(wq, [row0, 0]), s_addr, M, N, K, group, s_stride)


def gemm_i4(a: Buffer, wq: Buffer, s: Buffer | None, out: Buffer, group: int = 128, row0=0, rows: int | None = None,
            scales_major: str | None = None, scales_row: int | None = None):
    """`row0`/`rows`: the block of weight rows (and output columns) this workload covers, so one streamed chunk
    of many rows is consumed by several workloads under the channel limit (rows x K / group <= 1024).
    `scales_major`: 'row' ([N, K/group]) or 'group' ([K/group, N]); inferred from the shape when unambiguous.
    `scales_row`: the scales ride in `wq` itself (s=None): N = scales_row weight rows, then the group-major fp16
    scales [K/group, N] packed in whole rows -- one DMA per chunk instead of two (the engine's fixed cost per
    job, ~2.5 us, was 180 us per port in the MoE's 72 scale fills; M3)."""
    return _gemm_q("i4g", 0.5, a, wq, s, out, group, row0, rows, scales_major, scales_row)


def gemm_i8(a: Buffer, wq: Buffer, s: Buffer | None, out: Buffer, group: int = 128, row0=0, rows: int | None = None,
            scales_major: str | None = None, scales_row: int | None = None):
    """gemm_i4 with int8 weights (`wq` int8 [N, K], one byte per value); the lm_head's format."""
    return _gemm_q("i8g", 1.0, a, wq, s, out, group, row0, rows, scales_major, scales_row)
