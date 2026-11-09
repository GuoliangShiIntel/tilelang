//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Vendored verbatim from the VPUX plugin (applications.ai.vpu-accelerators.vpux-plugin),
// sw_runtime_kernels/kernels/src/asm/40xx/dequant_dynamic.asm: asm_dequant_dynamic_i4_f16_v0 only -- the entry the
// production kernel calls for the Nx128-by-Nx1 case (src/dynamic_dequantize.cpp:493, descrip/dynamic_dequantize_5000.txt).
// Called by tl_npu_dequant_i4_block (src/tl_templates/intel_npu/npu_rt_device.hpp).

//#########################################################################
// void asm_dequant_dynamic_i4_f16_v0(i4* inp, half* scl, half* out, int n)
//#########################################################################
// Scenario: inner broadcast
//   input: Nx128 xi4
//   scale: Nx1   xf16
//   zp   : none

.set pIn  i18
.set pScl i17
.set pOut i16
.set num  i15

.set runs i14
.set tmp  i13

.set off_00 i0
.set off_20 i1
.set off_40 i2
.set off_60 i3
.set off_02 i4

.set vScl0 v16 // for unroll[0]
.set vScl1 v17 // for unroll[1]

.code .text.asm_dequant_dynamic_i4_f16_v0
.align 16
.type asm_dequant_dynamic_i4_f16_v0, @function
asm_dequant_dynamic_i4_f16_v0:

    lsu0.ldil off_00, 0x00 || iau.add runs, num, 1
    lsu0.ldil off_02, 0x02 || iau.shr.u32 runs, runs, 1
    lsu0.ldil off_20, 0x20 || lsu1.ldil tmp, L_deq_dyn_v0_end
    lsu0.ldil off_40, 0x40 || lsu1.ldih tmp, L_deq_dyn_v0_end
    lsu0.ldil off_60, 0x60

    //-------- take off (hot-loop does 2x unrolls: inp=2x128 by scale=2x1)
    lsu0.ldx.512.i4.i8 v0, pIn, off_00 || lsu1.ldx.512.i4.i8 v2, pIn, off_20
    lsu0.ldx.512.i4.i8 v4, pIn, off_40 || lsu1.ldx.512.i4.i8 v6, pIn, off_60     || iau.incs pIn, +128
    lsu0.ld.512.16.R   vScl0, pScl     || lsu1.ldx.512.16.R  vScl1, pScl, off_02 || iau.incs pScl, +4
    nop 6


    cmu.cp.512.i8.f16 v0, v0.0 || iau.incs pOut, -512 // cancel 1st iter increment
    cmu.cp.512.i8.f16 v1, v0.1
 //---------------- hot loop (does 2x128 by 2x1)
  .lalign
  L_deq_dyn_v0_start:
    cmu.cp.512.i8.f16 v2, v2.0 || vau.mul.f16  v8, v0, vScl0 || lsu0.ldx.512.i4.i8 v0, pIn, off_00 || lsu1.ldx.512.i4.i8 v2, pIn, off_20 || bru.rpl tmp, runs, L_deq_dyn_v0_end
    cmu.cp.512.i8.f16 v3, v2.1 || vau.mul.f16  v9, v1, vScl0 || lsu0.ldx.512.i4.i8 v4, pIn, off_40 || lsu1.ldx.512.i4.i8 v6, pIn, off_60 || iau.incs pIn,  +128
    cmu.cp.512.i8.f16 v4, v4.0 || vau.mul.f16 v10, v2, vScl0 || lsu0.ld.512.16.R    vScl0, pScl || lsu1.ldx.512.16.R   vScl1, pScl, off_02 || iau.incs pOut, +512
    cmu.cp.512.i8.f16 v5, v4.1 || vau.mul.f16 v11, v3, vScl0 || lsu0.sto.256  v8.0, pOut, 0x000 || lsu1.sto.256  v8.1, pOut, 0x020 || iau.incs pScl, +4
  L_deq_dyn_v0_end: // Note: 1st unroll always executes, 2nd unroll only if 'num' was EVEN number thus predicate just 2nd unroll WRs
    cmu.cp.512.i8.f16 v6, v6.0 || vau.mul.f16 v12, v4, vScl1 || lsu0.sto.256  v9.0, pOut, 0x040 || lsu1.sto.256  v9.1, pOut, 0x060
    cmu.cp.512.i8.f16 v7, v6.1 || vau.mul.f16 v13, v5, vScl1 || lsu0.sto.256 v10.0, pOut, 0x080 || lsu1.sto.256 v10.1, pOut, 0x0A0
                                  vau.mul.f16 v14, v6, vScl1 || lsu0.sto.256 v11.0, pOut, 0x0C0 || lsu1.sto.256 v11.1, pOut, 0x0E0 || iau.incs num, -1
                                  vau.mul.f16 v15, v7, vScl1 || lsu0.sto.256 v12.0, pOut, 0x100 || lsu1.sto.256 v12.1, pOut, 0x120 || peu.pcix GT, 0x30
                                                                lsu0.sto.256 v13.0, pOut, 0x140 || lsu1.sto.256 v13.1, pOut, 0x160 || peu.pcix GT, 0x30
    cmu.cp.512.i8.f16 v0, v0.0 ||                               lsu0.sto.256 v14.0, pOut, 0x180 || lsu1.sto.256 v14.1, pOut, 0x1A0 || peu.pcix GT, 0x30
    cmu.cp.512.i8.f16 v1, v0.1 ||                               lsu0.sto.256 v15.0, pOut, 0x1C0 || lsu1.sto.256 v15.1, pOut, 0x1E0 || peu.pcix GT, 0x30 || iau.incs num, -1

 //-------- return
    bru.jmp i30
    nop 6

.unsetall

.asm_dequant_dynamic_i4_f16_v0_END:
.size asm_dequant_dynamic_i4_f16_v0, . - asm_dequant_dynamic_i4_f16_v0
