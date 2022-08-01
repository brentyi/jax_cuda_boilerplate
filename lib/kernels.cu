// CUDA kernel implementation.

#include <stdexcept>

#include "kernels.h"
#include "utils.h"

namespace cuda_ray_jax {

// Kernel implementations.
namespace {

template <typename Scalar>
__global__ void raysample_kernel(
    utils::NDArray<const float, 2> origins,     // (num_rays, xyz)
    utils::NDArray<const float, 2> directions,  // (num_rays, xyz)
    utils::NDArray<float, 3> points             // (num_rays, num_samples, xyz)
) {
    auto num_samples = points.shape[1];
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 10) points(0, 0, i) = i;
}

}  // namespace

// Host-side API.
__host__ void cuda_raysample_f32(cudaStream_t stream, void **buffers,
                                 const char *opaque, size_t opaque_len) {
    const auto *shapes = utils::ShapesDescriptor::from_bytes(opaque);
    const utils::ArrayBundle arrays{buffers, shapes};

    const auto [origins_shape, origins_rank] = arrays.shapes->get(0);
    assert(origins_rank == 2);
    const auto num_rays = origins_shape[0];

    // Compute our grid and block sizes.
    //
    // The block size should be a multiple of our warp size (32) and less than
    // the limit of 1024.
    //
    //
    const int grid_dim = 1024;
    const int block_dim = 128;
    raysample_kernel<float><<<grid_dim, block_dim, 0, stream>>>(
        {arrays, 0}, {arrays, 1}, {arrays, 2});

    utils::ThrowIfError(cudaGetLastError());
}

}  // namespace cuda_ray_jax
