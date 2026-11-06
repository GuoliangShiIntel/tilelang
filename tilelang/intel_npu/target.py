"""Intel NPU target handling: aliases and canonical attributes.

The TVM target kind ``intel_npu`` is registered natively in
``src/intel_npu/target.cc``. This module only maps user spellings onto it; it
performs no hardware detection, because the device is driven by an external
runner rather than by this process.
"""

from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_normalizer

TARGET_KIND = "intel_npu"
DEFAULT_ARCH = "npu5"

# User spellings accepted for the NPU5 (Panther Lake) platform.
_ARCH_ALIASES: dict[str, str] = {
    "intel_npu": DEFAULT_ARCH,
    "npu5": DEFAULT_ARCH,
    "ptl": DEFAULT_ARCH,
}


def make_intel_npu_target(arch: str = DEFAULT_ARCH, **attrs: object) -> Target:
    """Build an ``intel_npu`` Target with the given architecture and attributes."""
    config: dict[str, object] = {"kind": TARGET_KIND, "arch": arch}
    config.update(attrs)
    return Target(config)


def target_is_intel_npu(target: Target) -> bool:
    return target.kind.name == TARGET_KIND


def normalize_intel_npu_target(target: TargetLike) -> Target | None:
    """Map alias strings onto the canonical target; return None for anything else."""
    if not isinstance(target, str):
        return None
    spelling = target.strip()
    arch = _ARCH_ALIASES.get(spelling)
    if arch is None:
        return None
    return make_intel_npu_target(arch)


register_target_normalizer(TARGET_KIND, normalize_intel_npu_target, override=True)
