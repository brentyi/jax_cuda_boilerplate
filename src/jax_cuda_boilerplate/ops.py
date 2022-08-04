"""Operation definition for custom kernels."""


from typing import Tuple

import jax
from jax import core
from jax import numpy as jnp

from .jax_cuda_utils import make_cuda_primitive

# Define our primitive.
_raysample_prim = make_cuda_primitive("cuda_raysample_f32")


@_raysample_prim.def_abstract_eval
def _raysample_abstract_eval(
    origins: core.ShapedArray,
    directions: core.ShapedArray,
    *,
    num_samples: int,
) -> Tuple[core.ShapedArray]:
    assert origins.dtype == directions.dtype == jnp.float32
    assert origins.shape == directions.shape
    num_rays, dim = origins.shape  # type: ignore
    assert dim == 3
    return (core.ShapedArray((num_rays, num_samples, 3), origins.dtype),)


# Wrap.
def raysample(
    origins: jax.Array,
    directions: jax.Array,
    num_samples: int = 10,
) -> jnp.ndarray:
    """Sample some points along a set of 3D rays."""
    N = origins.shape[0]
    assert origins.shape == (N, 3)
    assert directions.shape == (N, 3)
    (out,) = _raysample_prim.bind(origins, directions, num_samples=num_samples)
    return out


def raysample_no_cuda(
    origins: jnp.ndarray,
    directions: jnp.ndarray,
    num_samples: int = 10,
) -> jnp.ndarray:
    """Same as raysample(), but implemented without our custom CUDA kernel."""
    N = origins.shape[0]
    assert origins.shape == (N, 3)
    assert directions.shape == (N, 3)

    ts = jnp.arange(num_samples) * 0.1
    return origins[:, None, :] + ts[None, :, None] * directions[:, None, :]
