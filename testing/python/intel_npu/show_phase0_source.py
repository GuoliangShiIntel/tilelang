"""Print the Phase 0 scalar kernel's generated SHAVE C++ (no hardware, no tile ops).
Usage: python testing/python/intel_npu/show_phase0_source.py"""

import tilelang
import tilelang.intel_npu.language as T


@T.prim_func
def main(A: T.Tensor((8, 16), T.float16)):
    with T.Kernel(2, threads=2) as tile:
        shave = T.get_thread_binding()
        for i in T.serial(4):
            A[tile * 2 + shave, i] = T.cast(0, T.float16)


if __name__ == "__main__":
    print(tilelang.lower(main, target="intel_npu").kernel_source)
