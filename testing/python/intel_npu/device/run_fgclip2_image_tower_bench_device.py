"""FG-CLIP2 image-tower real-device benchmark, at the SAME patch-count buckets npu_dsl's own
`dsl_bench.py`/`patch_sweep.py` measured (RUN_AND_RESULTS.md sections 8.4/8.6/8.7): 128, 256, 576,
784, 1024, 2048, 3072 (npu_dsl's own n_pad: 128, 256, 640, 896, 1024, 2048, 3072 -- this script's
BQ=64/C=128 row-padding naturally reproduces the same n_pad values).

27 real encoder layers (`make_fgclip2_encoder_layers_ddr`, real pretrained weights via a device
address table, the same pattern as `run_prefill_model_device.py`) + the attention-pooling head
(`make_fgclip2_pooling_head_ddr`) in ONE session, two launches. SYNTHETIC patch embeddings (seeded
random, matching npu_dsl's own dsl_bench.py methodology: "a seeded synthetic image patchified the
same way" -- timing does not depend on patch CONTENT, only shape, and real per-size images would
need host-side position-embedding resizing this script does not do). Real packed weights (shape-
and value-correct) throughout. No golden check at sizes other than 972 (no golden file exists for
synthetic patch counts); correctness at the shapes this backend's kernels actually claim to handle
is already covered by `run_fgclip2_layer0_device.py` (972 patches, real image, rel err 0.73%) and
the host-sim full-pipeline test. This script is a TIMING comparison only.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_fgclip2_image_tower_bench_device.py --valid 1024 [--no-build] [--trials 10]
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import tvm  # noqa: E402

import runner  # noqa: E402
import lowering  # noqa: E402
from fgclip2_kernels import FGCLIP2_TABLE_COLS, make_fgclip2_encoder_layers_ddr, make_fgclip2_pooling_head_ddr  # noqa: E402
from fgclip2_pack import Checkpoint, pack_layer, pack_vision_head  # noqa: E402

MODEL_ROOT = Path(r"C:\Intel\guoliang\dsl\fg-clip2-so400m")
HIDDEN, HEADS, HD_PAD, INTER_PAD, N_LAYERS = 1152, 16, 80, 4320, 27
BR, TN, C = 32, 32, 128
BQ_LAYERS, BQ_HEAD = 64, 16
QW = HEADS * HD_PAD
TILES, SHAVES = 1, 2
COL = {c: i for i, c in enumerate(FGCLIP2_TABLE_COLS)}


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.hw_sync_grid": False}):
        return lowering.lower(func)


def _padded_rows(valid: int) -> int:
    rows = -(-valid // BQ_LAYERS) * BQ_LAYERS
    return -(-rows // C) * C


def main(argv):
    if not MODEL_ROOT.exists():
        print(f"real checkpoint not found ({MODEL_ROOT})")
        return 2
    valid = arg(argv, "--valid", 1024)
    trials = arg(argv, "--trials", 10)
    rows = _padded_rows(valid)
    print(f"valid={valid} rows(n_pad)={rows}")

    layers_func = make_fgclip2_encoder_layers_ddr(N_LAYERS, rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD,
                                                  BR=BR, TN=TN, BQ=BQ_LAYERS, C=C)
    head_func = make_fgclip2_pooling_head_ddr(rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD,
                                              BR=BR, TN=TN, BQ=BQ_HEAD, C=C)
    low_layers = _lower(layers_func)
    low_head = _lower(head_func)
    name_layers = f"tl_fgclip2_image_layers_{rows}"
    name_head = f"tl_fgclip2_image_head_{rows}"

    build_dir = Path("X:/build") if Path("X:/").exists() else None
    if "--no-build" in argv:
        runner.write_kernel(low_layers.source, name_layers)
        runner.write_kernel(low_head.source, name_head)
    else:
        t0 = time.time()
        runner.build_all({name_layers: low_layers.source, name_head: low_head.source}, build_dir=build_dir)
        print(f"build {time.time() - t0:.0f} s")

    exe = (build_dir or (HERE / "build")) / "Release" / "npu_run.exe"
    s = runner.Session(exe, HERE / "run_fgclip2_image_tower_bench", TILES, SHAVES)
    print(f"session up, TURBO {s.turbo}")

    ck = Checkpoint(MODEL_ROOT)
    t0 = time.time()
    table = np.zeros((N_LAYERS, len(FGCLIP2_TABLE_COLS)), np.int64)
    for li in range(N_LAYERS):
        w = pack_layer(ck, "vision_model", li, hd_pad=HD_PAD, inter_pad=INTER_PAD)
        for name, key in (("Ln1W", "ln1w"), ("Ln1B", "ln1b"), ("Wq", "wq"), ("Bq", "bq"), ("Wk", "wk"), ("Bk", "bk"),
                          ("Wv", "wv"), ("Bv", "bv"), ("Wo", "wo"), ("Bo", "bo"), ("Ln2W", "ln2w"), ("Ln2B", "ln2b"),
                          ("W1", "w1"), ("B1", "b1"), ("W2", "w2"), ("B2", "b2")):
            s.upload(f"L{li}_{name}", np.ascontiguousarray(w[key]))
            table[li, COL[name]] = s.address(f"L{li}_{name}")
    s.upload("Table", table)
    head_w = pack_vision_head(ck, hd_pad=HD_PAD, inter_pad=INTER_PAD)
    print(f"weights uploaded in {time.time() - t0:.0f} s (27 layers + head)")

    rng = np.random.default_rng(0)
    embed = (rng.standard_normal((valid, HIDDEN)) * 0.3).astype(np.float32)
    hio = np.zeros((rows, HIDDEN), np.float32)
    hio[:valid] = embed
    s.upload("Hio", hio)
    for nm, shape, dt in [("Xn", (rows, HIDDEN), np.float16), ("Qp", (rows, QW), np.float16),
                          ("Kp", (rows, QW), np.float16), ("Vp", (rows, QW), np.float16),
                          ("Ao", (rows, QW), np.float16), ("Act", (rows, INTER_PAD), np.float16)]:
        s.upload(nm, np.zeros(shape, dt))

    s.upload("PostLnW", head_w["post_ln_w"]); s.upload("PostLnB", head_w["post_ln_b"])
    s.upload("MapQ", np.tile(head_w["map_q"], (BQ_HEAD, 1)))
    s.upload("MapWk", head_w["map_wk"]); s.upload("MapBk", head_w["map_bk"])
    s.upload("MapWv", head_w["map_wv"]); s.upload("MapBv", head_w["map_bv"])
    s.upload("MapWo", head_w["map_wo"]); s.upload("MapBo", head_w["map_bo"])
    s.upload("HLnW", head_w["hln_w"]); s.upload("HLnB", head_w["hln_b"])
    s.upload("MapW1", head_w["map_w1"]); s.upload("MapB1", head_w["map_b1"])
    s.upload("MapW2", head_w["map_w2"]); s.upload("MapB2", head_w["map_b2"])
    s.upload("Ks", np.zeros((rows, QW), np.float16)); s.upload("Vs", np.zeros((rows, QW), np.float16))
    s.upload("HAo", np.zeros((BQ_HEAD, QW), np.float16))
    s.upload("ALn", np.zeros((BQ_HEAD, HIDDEN), np.float16))
    s.upload("Act1", np.zeros((BQ_HEAD, INTER_PAD), np.float16))
    s.upload("Emb", np.zeros((BQ_HEAD, HIDDEN), np.float32))

    s.kernel("layers", name_layers)
    s.kernel("head", name_head)

    by_name_layers = {"Hio": "Hio", "Table": "Table", "Xn": "Xn", "Qp": "Qp", "Kp": "Kp", "Vp": "Vp", "Ao": "Ao", "Act": "Act"}
    by_name_head = {"Hio": "Hio", "PostLnW": "PostLnW", "PostLnB": "PostLnB", "MapQ": "MapQ", "MapWk": "MapWk",
                    "MapBk": "MapBk", "MapWv": "MapWv", "MapBv": "MapBv", "MapWo": "MapWo", "MapBo": "MapBo",
                    "HLnW": "HLnW", "HLnB": "HLnB", "MapW1": "MapW1", "MapB1": "MapB1", "MapW2": "MapW2",
                    "MapB2": "MapB2", "Xn": "Xn", "Ks": "Ks", "Vs": "Vs", "Ao": "HAo", "ALn": "ALn",
                    "Act1": "Act1", "Emb": "Emb"}
    layers_bufs = [by_name_layers[n] for n in low_layers.param_names]
    head_bufs = [by_name_head[n] for n in low_head.param_names]

    layers_ms, head_ms = [], []
    for t in range(trials):
        host1, kernel1 = s.launch("layers", *layers_bufs)
        host2, kernel2 = s.launch("head", *head_bufs)
        layers_ms.append(host1)
        head_ms.append(host2)
        print(f"trial {t}: layers {host1:.1f} ms (kernel {kernel1:.1f}), head {host2:.1f} ms (kernel {kernel2:.1f})")

    emb = s.download("Emb", np.float32, (BQ_HEAD, HIDDEN))[0]
    finite = bool(np.all(np.isfinite(emb)))
    layers_ms.sort(); head_ms.sort()
    lm, hm = layers_ms[len(layers_ms) // 2], head_ms[len(head_ms) // 2]
    print(f"\n@@RESULT@@ valid={valid} rows={rows} median_layers_ms={lm:.2f} median_head_ms={hm:.2f} "
          f"median_total_ms={lm + hm:.2f} finite_output={finite}")
    s.close()
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
