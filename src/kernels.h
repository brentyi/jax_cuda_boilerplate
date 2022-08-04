#pragma once

#include <cuda_runtime_api.h>

namespace jax_cuda_boilerplate {

void cuda_raysample_f32(cudaStream_t stream, void **buffers,
                        const char *backend_config,
                        std::size_t backend_config_len);

}  // namespace jax_cuda_boilerplate
