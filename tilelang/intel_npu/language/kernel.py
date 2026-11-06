"""Intel NPU dialect of ``T.Kernel``: a persistent 1-D grid of tiles, each running its SHAVEs."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *tiles: int | tirx.PrimExpr,
    threads: int | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for the Intel NPU.

    ``blockIdx.x`` is the tile (cluster) and ``threadIdx.x`` the SHAVE within
    it. The grid is persistent: it names the tiles the kernel occupies, and
    work is dealt over them by loops in the body. One ``prim_func`` is one
    ``kernel_entry`` and one launch, however many phases it holds.

    Parameters
    ----------
    *tiles : int | PrimExpr
        Exactly one extent: the number of tiles. Multi-dimensional grids have
        no meaning on this target and are rejected.
    threads : int, optional
        SHAVEs per tile. Defaults to the target's ``shaves_per_tile``.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(num_tiles, threads=2) as tile:
            ...
    """
    if len(tiles) != 1:
        raise ValueError(f"intel_npu: T.Kernel takes exactly one grid extent (the tile count); got {len(tiles)} dimensions")
    return launch_kernel(tiles, threads=threads)
