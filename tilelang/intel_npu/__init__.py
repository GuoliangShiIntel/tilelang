"""Intel NPU (NPU5 / Panther Lake) backend package.

Import order is cycle-safe: target registration, native transforms, the
language dialect, then the pipeline/codegen/execution policy the manifest
binds together.
"""

from . import target as target  # noqa: F401
from . import transform as transform  # noqa: F401
from . import language as language  # noqa: F401
from . import pipeline as pipeline  # noqa: F401
from . import codegen as codegen  # noqa: F401
from . import execution_backend as execution_backend  # noqa: F401
from . import backend as backend  # noqa: F401
