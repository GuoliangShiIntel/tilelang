"""Phase 3a device check: sync_grid on hardware barriers vs the software rendezvous.

Kernel: `rounds` grid barriers per launch with a DDR token chain that is only correct when every
barrier holds (kernels.make_sync_chain). Builds and times two arms, interleaved and UNPROFILED:
  hw  : tl.intel_npu.hw_sync_grid=True  (3 rotating hardware barrier ids at the top of the pool)
  sw  : tl.intel_npu.hw_sync_grid=False (toolkit npu::Sync::waitGlobal)
Prints per-arm host-wall medians and the implied per-barrier cost difference; exit 0 when both
arms are correct. --nosync builds the chain without barriers as the negative control (must FAIL).

Usage: python run_sync_chain_device.py [--rounds N] [--repeat R] [--nosync]
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import runner  # noqa: E402
import lowering  # noqa: E402
from kernels import make_sync_chain  # noqa: E402
from tilelang import tvm  # noqa: E402
import tilelang.intel_npu.language as T  # noqa: E402

TILES, SHAVES = 3, 2


def make_nosync_chain(rounds):
    W = TILES * SHAVES
    LINE = 4

    @T.prim_func
    def sync_chain_nosync(Tok: T.Tensor((rounds + 1, W * LINE), T.int32)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            shave = T.get_thread_binding()
            worker = tile * SHAVES + shave
            for r in T.serial(rounds):
                Tok[r + 1, worker * LINE] = Tok[r, ((worker + 1) % W) * LINE] + 1

    return sync_chain_nosync


def build_arm(name, rounds, hw_sync, nosync=False):
    config = {"tl.intel_npu.hw_sync_grid": hw_sync, "tl.intel_npu.derive_global_sync": not nosync}
    with tvm.transform.PassContext(config=config):
        low = lowering.lower(make_nosync_chain(rounds) if nosync else make_sync_chain(rounds, TILES, SHAVES))
    assert ("sync_barrier_base" in low.source) == hw_sync
    return runner.build(low.source, name), low


def run_arm(db, rounds, repeat):
    W = TILES * SHAVES
    tok = np.zeros((rounds + 1, W * 4), np.int32)
    res = runner.run(db, [("inout", tok)], TILES, SHAVES, HERE / f"run_{db.kernel}", repeat=repeat, warmup=2)
    runner.check_status(res.status)
    last = tok[rounds, ::4]
    ok = bool(np.all(last == rounds))
    return res, ok, last.tolist()


def main(argv):
    rounds = int(argv[argv.index("--rounds") + 1]) if "--rounds" in argv else 200
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 10
    if "--nosync" in argv:
        db, _ = build_arm("tl_sync_chain_nosync", rounds, hw_sync=True, nosync=True)
        res, ok, last = run_arm(db, rounds, repeat)
        print(f"nosync: last row {last} -> {'PASS' if ok else 'FAIL'}; negative control " + ("behaved" if not ok else "BROKEN"))
        return 0 if not ok else 1
    hw, _ = build_arm("tl_sync_chain_hw", rounds, hw_sync=True)
    sw, _ = build_arm("tl_sync_chain_sw", rounds, hw_sync=False)
    meds = {"hw": [], "sw": []}
    oks = {"hw": True, "sw": True}
    for i in range(3):  # interleaved rounds
        for name, db in (("hw", hw), ("sw", sw)):
            res, ok, last = run_arm(db, rounds, repeat)
            meds[name].append(res.median_ms)
            oks[name] &= ok
            print(f"round {i} {name}: median {res.median_ms:.3f} ms over {repeat} launches; correct={ok}")
    mh, ms = float(np.median(meds["hw"])), float(np.median(meds["sw"]))
    print(f"\n{rounds} barriers/launch, UNPROFILED host wall incl. submit: hw {mh:.3f} ms, sw {ms:.3f} ms; "
          f"per barrier hw-sw = {(mh - ms) / rounds * 1000:.2f} us (negative = hw faster)")
    print(f"correct: hw={oks['hw']} sw={oks['sw']}; kernels hw {hw.source_sha256[:12]} sw {sw.source_sha256[:12]}")
    return 0 if oks["hw"] and oks["sw"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
