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

__all__ = ["gemm_i4", "gemm_i8", "dequant_i4_gm"]


def _addr(buf: Buffer, idx=None):
    return tirx.address_of(tirx.BufferLoad(buf, idx if idx is not None else [0] * len(buf.shape)))


def dequant_i4_gm(out: Buffer, packed: Buffer, scales: Buffer, group: int = 128,
                  packed_row=0, scales_col=0):
    """Expand packed int4 CMX rows using group-major fp16 scales; retain explicit IR dependencies."""
    rows, width = (int(dim) for dim in out.shape)
    group = int(group)
    if out.dtype != "float16" or packed.dtype != "uint8" or scales.dtype != "float16":
        raise ValueError("intel_npu: dequant_i4_gm requires fp16 output/scales and uint8 packed weights")
    if (group <= 0 or width % group or int(packed.shape[1]) != width // 2
            or int(scales.shape[0]) != width // group or int(packed.shape[0]) < rows
            or int(scales.shape[1]) < rows):
        raise ValueError("intel_npu: dequant_i4_gm has incompatible weight/scale shapes")
    if isinstance(packed_row, int) and not 0 <= packed_row <= int(packed.shape[0]) - rows:
        raise ValueError("intel_npu: dequant_i4_gm packed row offset is out of bounds")
    if isinstance(scales_col, int) and not 0 <= scales_col <= int(scales.shape[1]) - rows:
        raise ValueError("intel_npu: dequant_i4_gm scale column offset is out of bounds")
    if any(buf.scope() not in ("local", "shared", "shared.dyn") for buf in (out, packed, scales)):
        raise ValueError("intel_npu: dequant_i4_gm operands must live in CMX")
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_dequant_i4_gm"), _addr(out),
                           _addr(packed, [packed_row, 0]), _addr(scales, [0, scales_col]),
                           rows, width, group, int(scales.shape[1]))


def _gemm_q(op_name: str, bytes_per_k: float, a: Buffer, wq: Buffer, s: Buffer | None, out: Buffer, group: int, row0, rows,
            scales_major: str | None, scales_row: int | None, scales_at=None, defer: bool = False):
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
        # group-major scales may be a WINDOW of a larger buffer -- a wide table of several chunks' scales, or a
        # stack of per-chunk [K/group, N] blocks -- so that one DMA serves many chunks. `scales_at` names this
        # chunk's origin in it and the buffer's own second extent is the pitch; it is usually a loop variable, so
        # only the pitch is checkable here.
        group_major = len(s_shape) == 2 and (s_shape == [ng, N_all] or scales_at is not None)
        if scales_at is not None and (len(s_shape) != 2 or s_shape[1] < N_all and s_shape[1] != int(wq.shape[0])):
            raise ValueError(f"intel_npu: {op_name}: scales_at needs a 2-D group-major scales buffer, got {s_shape}")
        if scales_major is None:
            if row_major and group_major:
                raise ValueError(f"intel_npu: {op_name}: scales {s_shape} could be row- or group-major (N == K/group); "
                                 f"say scales_major='row' or 'group'")
            if not (row_major or group_major):
                raise ValueError(f"intel_npu: {op_name}: scales {s_shape} are neither [N, K/group] = [{N_all}, {ng}] "
                                 f"nor [K/group, >= N]")
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
        at = [0, row0] if scales_at is None else [scales_at[0], scales_at[1] + row0]
        s_addr, s_stride = _addr(s, at), int(s.shape[1])
    if defer and scales_row is None and scales_major == "row":
        raise ValueError(f"intel_npu: {op_name}: defer needs group-major scales read in place (the transposed copy does not outlive the call)")
    return tirx.call_intrin("handle", tirx.op.Op.get(f"tl.intel_npu_dpu_matmul_{op_name}"),
                            _addr(out, [0, row0]), _addr(a), _addr(wq, [row0, 0]), s_addr, M, N, K, group, s_stride, 1 if defer else 0)


def gemm_i4(a: Buffer, wq: Buffer, s: Buffer | None, out: Buffer, group: int = 128, row0=0, rows: int | None = None,
            scales_major: str | None = None, scales_row: int | None = None, scales_at=None, defer: bool = False):
    """`row0`/`rows`: the block of weight rows (and output columns) this workload covers, so one streamed chunk
    of many rows is consumed by several workloads under the channel limit (rows x K / group <= 1024).
    `scales_major`: 'row' ([N, K/group]) or 'group' ([K/group, N]); inferred from the shape when unambiguous.
    `scales_at`: `(row, col)` of this chunk's scales inside a larger group-major buffer, read in place at that
    buffer's pitch. One DMA then serves many chunks, which matters because the engine's fixed cost per job (~2.5
    us) multiplies by the chunk count: a MoE expert fetches a wide `[K/group, items x N]` table once, and the
    lm_head fetches a run of per-chunk `[K/group, N]` blocks once.
    `scales_row`: the scales ride in `wq` itself (s=None): N = scales_row weight rows, then the group-major fp16
    scales [K/group, N] packed in whole rows -- one DMA per chunk instead of two, at the cost of putting a
    scheduling constant into the weight file (the layout prefill cannot share).
    `defer`: the SHAVE's group reduction of this workload runs during the NEXT quantised gemm's DPU time instead of
    right after this one's (npu_dsl fused_rt's defer); legal when nothing reads `out[:, row0:row0+rows]` before that
    next gemm (the runtime also flushes a pending reduction at a store, a sync, a ring release or fill, and at the
    end). The pattern: all but the last of the workloads that consume one streamed chunk."""
    return _gemm_q("i4g", 0.5, a, wq, s, out, group, row0, rows, scales_major, scales_row, scales_at, defer)


def gemm_i8(a: Buffer, wq: Buffer, s: Buffer | None, out: Buffer, group: int = 128, row0=0, rows: int | None = None,
            scales_major: str | None = None, scales_row: int | None = None, scales_at=None, defer: bool = False):
    """gemm_i4 with int8 weights (`wq` int8 [N, K], one byte per value); the lm_head's format."""
    return _gemm_q("i8g", 1.0, a, wq, s, out, group, row0, rows, scales_major, scales_row, scales_at, defer)
