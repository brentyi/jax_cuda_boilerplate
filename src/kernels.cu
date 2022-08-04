// Toy CUDA kernel implementation.

#include <iostream>
#include <stdexcept>

#include "jax_utils.h"
#include "kernels.h"

namespace jax_cuda_boilerplate {

// Kernel implementations.
namespace {
// Sample points along some rays.
template <typename Scalar>
__global__ void raysample_kernel(
    jax_utils::DeviceArray<const float, 2> origins,  // (num_rays, xyz)
    jax_utils::DeviceArray<const float, 2> dirs,     // (num_rays, xyz)
    jax_utils::DeviceArray<float, 3> out  // (num_rays, num_samples, xyz)
) {
    const int ray_idx = blockIdx.x;
    const int sample_idx = threadIdx.x;

    // Check bounds.
    const int num_rays = out.shape[0];
    const int num_samples = out.shape[1];
    if (ray_idx >= num_rays || sample_idx >= num_samples) return;

    // Grab pointers to each input/output vector.
    const auto o = origins(ray_idx).assert_shape(3);
    const auto d = dirs(ray_idx).assert_shape(3);
    const auto sample = out(ray_idx, sample_idx).assert_shape(3);

    // Write sample.
    float t = sample_idx * 0.1;
    sample(0) = o(0) + t * d(0);
    sample(1) = o(1) + t * d(1);
    sample(2) = o(2) + t * d(2);
}

}  // namespace

// Host-side API.
__host__ void cuda_raysample_f32(cudaStream_t stream, void **buffers,
                                 const char *backend_config,
                                 size_t backend_config_len) {
    assert(backend_config_len == sizeof(jax_utils::ShapesDescriptor));
    const auto *shapes =
        jax_utils::ShapesDescriptor::from_bytes(backend_config);
    const jax_utils::ArrayBundle arrays{buffers, shapes};

    // Output array should have shape (num_rays, num_samples, 3).
    const int *out_shape = arrays.shapes->get_shape(2);
    const int out_rank = arrays.shapes->ranks[2];
    assert(out_rank == 3);
    assert(out_shape[2] == 3);

    // Super rudimentary grid/block-size logic: we'll make each block
    // responsible for one ray, and each thread responsible for one sample.
    int grid_size = out_shape[0];
    int block_size = out_shape[1];

    raysample_kernel<float><<<grid_size, block_size, 0, stream>>>(
        {arrays, 0}, {arrays, 1}, {arrays, 2});

    jax_utils::ThrowIfError(cudaGetLastError());
}

}  // namespace jax_cuda_boilerplate
