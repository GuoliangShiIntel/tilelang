"""The compressed-tensors (pack-quantized) checkpoint of a Qwen3.5-MoE-family model, read lazily with safetensors.

Format (config.json `quantization_config`; the same as npu_dsl's `qwen36_e2e/reference/checkpoint.py` documents):
  <prefix>.weight_packed int32 [N, K * bits / 32]   little-endian nibbles / bytes in element order, stored as v + 2^(bits-1)
  <prefix>.weight_scale  bf16  [N, K / 128]
  <prefix>.weight_shape  [N, K]
  W[n, k] = v[n, k] * scale[n, k // 128]
group_0 (every Linear) is int4; group_1 (lm_head) is int8. Not quantised (bf16): embed_tokens, the router (mlp.gate),
shared_expert_gate, in_proj_a / in_proj_b, the norms, conv1d, A_log, dt_bias.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LM = "model.language_model."


def _bf16_to_f32(t) -> np.ndarray:
    """torch bf16 -> numpy fp32 without a torch dependency at the call sites that only need numpy."""
    import torch
    return t.to(torch.float32).numpy()


@dataclass(frozen=True)
class Cfg:
    hidden: int
    layers: int
    layer_types: tuple
    eps: float
    vocab: int
    n_heads: int
    n_kv: int
    head_dim: int
    rot_dim: int
    rope_theta: float
    lin_kh: int
    lin_vh: int
    lin_dk: int
    lin_dv: int
    conv_k: int
    n_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int

    @classmethod
    def from_hf(cls, c: dict) -> "Cfg":
        rp = c["rope_parameters"]
        return cls(hidden=c["hidden_size"], layers=c["num_hidden_layers"], layer_types=tuple(c["layer_types"]),
                   eps=c["rms_norm_eps"], vocab=c["vocab_size"], n_heads=c["num_attention_heads"],
                   n_kv=c["num_key_value_heads"], head_dim=c["head_dim"],
                   rot_dim=int(c["head_dim"] * rp["partial_rotary_factor"]), rope_theta=rp["rope_theta"],
                   lin_kh=c["linear_num_key_heads"], lin_vh=c["linear_num_value_heads"],
                   lin_dk=c["linear_key_head_dim"], lin_dv=c["linear_value_head_dim"],
                   conv_k=c["linear_conv_kernel_dim"], n_experts=c["num_experts"], top_k=c["num_experts_per_tok"],
                   moe_inter=c["moe_intermediate_size"], shared_inter=c["shared_expert_intermediate_size"])

    @property
    def conv_dim(self) -> int:
        return 2 * self.lin_kh * self.lin_dk + self.lin_vh * self.lin_dv


class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        full = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        self.config = full.get("text_config", full)
        self.cfg = Cfg.from_hf(self.config)
        self.weight_map = json.loads((self.path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        q = full["quantization_config"]
        assert q["quant_method"] == "compressed-tensors" and q["format"] == "pack-quantized", q
        for g in q["config_groups"].values():
            w = g["weights"]
            assert w["symmetric"] and w["strategy"] == "group" and w["group_size"] == 128 and w["type"] == "int", w
        self.group = 128
        self._handles = {}

    def _handle(self, fname):
        h = self._handles.get(fname)
        if h is None:
            from safetensors import safe_open
            h = safe_open(str(self.path / fname), "pt")
            self._handles[fname] = h
        return h

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def raw(self, name: str):
        return self._handle(self.weight_map[name]).get_tensor(name)

    def f32(self, name: str) -> np.ndarray:
        return _bf16_to_f32(self.raw(name))

    def f16(self, name: str) -> np.ndarray:
        return self.f32(name).astype(np.float16)

    def quantized(self, prefix: str):
        """(v int8 [N, K] signed values, scales fp16 [N, K/128], bits)."""
        import torch
        packed = self.raw(prefix + ".weight_packed")
        n, k = (int(x) for x in self.raw(prefix + ".weight_shape").tolist())
        bits = packed.shape[1] * 32 // k
        assert packed.shape[0] == n and bits in (4, 8), (prefix, tuple(packed.shape), n, k)
        u = packed.contiguous().view(torch.uint8).numpy()   # [N, K * bits / 8], byte order = element order
        if bits == 8:
            v = (u.astype(np.int16) - 128).astype(np.int8)
        else:
            v = np.stack([u & 0xF, u >> 4], axis=-1).reshape(n, -1).astype(np.int16) - 8
            v = v.astype(np.int8)
        assert v.shape == (n, k), (v.shape, n, k)
        return v, self.f16(prefix + ".weight_scale"), bits

    def dense_f32(self, prefix: str) -> np.ndarray:
        """Any Linear as fp32 [N, K] (quantised ones dequantised): the oracle's view."""
        if self.has(prefix + ".weight_packed"):
            v, s, _ = self.quantized(prefix)
            return v.astype(np.float32) * np.repeat(s.astype(np.float32), self.group, axis=1)
        return self.f32(prefix + ".weight")


def layer_prefix(i: int) -> str:
    return f"{LM}layers.{i}."


def default_checkpoint() -> Path:
    p = os.environ.get("TL_NPU_CKPT")
    if not p:
        raise RuntimeError("TL_NPU_CKPT is not set (the model checkpoint directory)")
    return Path(p)
