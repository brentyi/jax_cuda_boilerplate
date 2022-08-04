from typing import Any

from jax import core

from . import custom_calls_ext as custom_calls_ext
from . import jax_utils_ext as jax_utils_ext

ir: Any
Value = Any

def make_cuda_primitive(custom_call_target: str) -> core.Primitive: ...
