#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cuda_ray_jax {

void cuda_raysample_f32(cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len);

}  // namespace cuda_ray_jax
