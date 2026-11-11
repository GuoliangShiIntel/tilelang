"""The fp32 CPU reference's answer for a prompt, saved once so the device run never hosts the reference next to
18 GB of pinned weights (the two together swap a 32 GB box). Model-side code: npu_dsl's `qwen36_e2e.reference`
(NPU_DSL_ROOT), which reads the same compressed-tensors checkpoint.

    python testing/python/intel_npu/e2e/reference_golden.py --ckpt DIR --out golden.npz [--prompt "..."|--ids ...]
                                                            [--new 16] [--layers N] [--expert-cache 64]

Saves: prompt_ids, out_ids (greedy), logits per step (fp16 [1 + new, V]: the prompt's last position then each
generated token's), and for the FIRST prompt token the per-layer trace (mix, moe, out: fp32 [layers, H] each).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from checkpoint import Checkpoint  # noqa: E402


def chat_ids(tok, prompt: str):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok.encode(text, add_special_tokens=False).ids


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("TL_NPU_CKPT"), required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Give me a short introduction to large language models.")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--expert-cache", type=int, default=64)
    ap.add_argument("--prefill-batch", type=int, default=1)
    a = ap.parse_args(argv)
    if a.prefill_batch < 1:
        ap.error("--prefill-batch must be positive")
    root = os.environ.get("NPU_DSL_ROOT")
    if not root:
        print("NPU_DSL_ROOT is not set (the libraries.npu.dsl checkout: its qwen36_e2e.reference is the oracle)")
        return 2
    sys.path.insert(0, root)
    import torch
    from tokenizers import Tokenizer
    from qwen36_e2e.reference.checkpoint import Checkpoint as RefCkpt
    from qwen36_e2e.reference.model import Qwen36Ref, State

    ck = Checkpoint(a.ckpt)
    layers = a.layers if a.layers is not None else ck.cfg.layers
    tok = Tokenizer.from_file(str(Path(a.ckpt) / "tokenizer.json"))
    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    elif a.raw:
        ids = tok.encode(a.prompt, add_special_tokens=False).ids
    else:
        ids = chat_ids(tok, a.prompt)
    ref = Qwen36Ref(RefCkpt(str(a.ckpt)), expert_cache=a.expert_cache)
    st = State()
    t0 = time.time()
    with torch.no_grad():
        trace = []
        logits = ref.forward(ids[:1], st, layers=layers, trace=trace)
        trace_mix = np.stack([t["mix"][0].numpy() for t in trace]).astype(np.float32)
        trace_moe = np.stack([t["moe"][0].numpy() for t in trace]).astype(np.float32)
        trace_out = np.stack([t["out"][0].numpy() for t in trace]).astype(np.float32)
        del trace
        for start in range(1, len(ids), a.prefill_batch):
            logits = ref.forward(ids[start:start + a.prefill_batch], st, layers=layers)
            if a.prefill_batch > 1:
                print(f"reference prompt {min(start + a.prefill_batch, len(ids))}/{len(ids)} tokens", flush=True)
        print(f"prefill {len(ids)} tokens: {time.time() - t0:.0f} s", flush=True)
        all_logits, out = [logits.numpy().astype(np.float16)], []
        for _ in range(a.new):
            t = int(torch.argmax(logits))
            out.append(t)
            print(tok.decode([t]), end="", flush=True)
            logits = ref.forward([t], st, layers=layers)
            all_logits.append(logits.numpy().astype(np.float16))
    print(f"\n[{len(out)} tokens, {time.time() - t0:.0f} s]")
    np.savez(a.out, prompt_ids=np.array(ids, np.int64), out_ids=np.array(out, np.int64), logits=np.stack(all_logits),
             trace_mix=trace_mix, trace_moe=trace_moe, trace_out=trace_out, layers=np.int64(layers))
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
