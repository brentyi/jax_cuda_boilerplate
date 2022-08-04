import time
from typing import Any, Callable, Tuple

import dcargs
import jax
import numpy as onp
import termcolor
from jax import numpy as jnp

from jax_cuda_boilerplate import ops


def get_median_runtime(func: Callable[[], Any], trials: int = 5000) -> float:
    """Call a function several times, and compute the median runtime."""
    times = []
    for i in range(trials):
        start_time = time.time()
        func()
        times.append(time.time() - start_time)
    return float(onp.median(times))


def main(
    num_rays_values: Tuple[int, ...] = (8192,),
    num_samples_values: Tuple[int, ...] = (128, 256, 512),
):
    """Compare our toy ray sampling implementation against a native JAX one. (the native
    JAX one will usually be faster!)

    Args:
        num_rays_values: Number of rays to batch for each run.
        num_samples_values: Number of samples to generate for each ray.
    """
    raysample = jax.jit(ops.raysample, static_argnames=("num_samples",))
    raysample_no_cuda = jax.jit(ops.raysample_no_cuda, static_argnames=("num_samples",))

    for num_rays in num_rays_values:
        prng_origins, prng_directions = jax.random.split(jax.random.PRNGKey(0))
        origins = jax.random.normal(prng_origins, shape=(num_rays, 3))
        directions = jax.random.normal(prng_directions, shape=(num_rays, 3))
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)

        print(f"{num_rays=}")
        for num_samples in num_samples_values:
            # JIT compile + check that outputs match.
            a = raysample_no_cuda(origins, directions, num_samples=num_samples)
            b = raysample(origins, directions, num_samples=num_samples)
            assert a.shape == b.shape
            onp.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

            # Get timings.
            no_cuda_time = get_median_runtime(
                lambda: raysample_no_cuda(
                    origins, directions, num_samples=num_samples
                ).block_until_ready()
            )
            cuda_time = get_median_runtime(
                lambda: raysample(
                    origins, directions, num_samples=num_samples
                ).block_until_ready()
            )

            percent_change = abs(cuda_time - no_cuda_time) / no_cuda_time * 100.0
            delta = f"{percent_change:.1f}%"
            if cuda_time < no_cuda_time:
                delta = termcolor.colored(f"-{delta}", color="green")
            else:
                delta = termcolor.colored(f"+{delta}", color="red")

            # Print.
            print(f"\t{num_samples=}")
            print(f"\t\tJust JAX (micros): \t {no_cuda_time * 1e6:.2f}")
            print(f"\t\tOur CUDA (micros): \t {cuda_time * 1e6:.2f} {delta}")
            print()


# ops.raysample_no_cuda
if __name__ == "__main__":
    dcargs.cli(main)
