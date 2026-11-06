"""Print the generated C++ for the Phase 2 two-phase kernel. Usage: python show_two_phase_source.py [nosync]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tilelang  # noqa: E402

from kernels import make_gemm_two_phase  # noqa: E402

if __name__ == "__main__":
    func = make_gemm_two_phase(256, 256, 512, with_sync="nosync" not in sys.argv[1:])
    print(tilelang.lower(func, target="intel_npu").kernel_source)
