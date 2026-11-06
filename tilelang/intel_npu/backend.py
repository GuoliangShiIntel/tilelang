"""Intel NPU backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline
from .target import TARGET_KIND

BACKEND = register_backend(
    BackendModule(
        name="intel_npu",
        target_kinds=(TARGET_KIND,),
        pipelines={TARGET_KIND: PassPipeline(TARGET_KIND, pipeline.IntelNpuPassPipelineBody)},
        device_codegens={TARGET_KIND: DeviceCodegen(TARGET_KIND, build_without_compile=codegen.build_source)},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
