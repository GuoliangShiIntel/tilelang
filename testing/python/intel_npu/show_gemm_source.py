"""Print the generated SHAVE C++ for the Phase 1 GEMM. Usage: python show_gemm_source.py [ksplit]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tilelang  # noqa: E402

from kernels import make_gemm, make_gemm_ksplit  # noqa: E402

if __name__ == "__main__":
    func = make_gemm_ksplit(64, 64, 256) if "ksplit" in sys.argv[1:] else make_gemm(256, 256, 512)
    print(tilelang.lower(func, target="intel_npu").kernel_source)
