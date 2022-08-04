// Helpers for doing custom CUDA stuff with JAX.
//
// Currently contains:
// - A descriptor object for passing.
// - A lightweight wrapper for indexing into N-dimensional arrays. Similar to
//   Eigen::TensorMap.
// - Error handling logic.

#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <type_traits>
#include <vector>

namespace jax_utils {

constexpr int MAX_TENSORS = 3;
constexpr int MAX_DIMENSIONS = 10;

// Simple descriptor for array shapes. This is designed to be simple to
// serialize and pass as the `opaque` parameter of our XLA custom calls. We
// might consider including more general metadata in the future, but just having
// all of the buffer shapes should cover most cases.
//
// As per the XLA documentation, we use a backend config/opaque parameter
// instead of putting shape information in a buffer because the shape
// information is often needed on the host to determine block and grid
// dimensions.
struct ShapesDescriptor {
    // Deserialization; this is just a pointer cast with some very basic error
    // checking.
    __host__ static const ShapesDescriptor *from_bytes(const char *bytes) {
        auto *out = reinterpret_cast<const ShapesDescriptor *>(bytes);
        assert(out->canary_ == 94709);
        return out;
    }

    __host__ inline const int *get_shape(int i) const {
        return &flattened_dimensions[0] + ranks_cumsum[i];
    }

    int count;
    std::array<int, MAX_TENSORS> ranks;
    std::array<int, MAX_TENSORS> ranks_cumsum;
    std::array<int, MAX_DIMENSIONS> flattened_dimensions;

   private:
    int canary_ = 94709;
};

// Structure for making the NDArray constructor more succinct.
struct ArrayBundle {
    void **buffers;
    const ShapesDescriptor *shapes;
};

// Light helper for indexing into dense N-dimensional arrays. Mirroring JAX, we
// assume everything is row-major.
//
// Designed to be instantiated on the host, then passed to the device.
template <typename scalar_t, int TRank>
class DeviceArray {
   public:
    // Default constructor.
    __host__ __device__ DeviceArray() {}

    // Standard constructor; computes shape and stride.
    __host__ DeviceArray(scalar_t *data, const int *shape, const int rank)
        : data(data) {
        assert(rank == TRank);

        int cumprod = 1;
        for (int i = 0; i < TRank; i++) {
            this->shape[i] = shape[i];
            this->strides[TRank - 1 - i] = cumprod;
            cumprod *= shape[TRank - 1 - i];
        }
    }

    // Initialization from an ArrayBundle.
    __host__ DeviceArray(const ArrayBundle &arrays, int i)
        : DeviceArray(reinterpret_cast<scalar_t *>(arrays.buffers[i]),
                      arrays.shapes->get_shape(i), arrays.shapes->ranks[i]) {}

    // Eigen-inspired indexing syntax. When we have fewer indices than the rank
    // of our array, we return a lower-dimensional slice of the array.
    template <
        typename... Ints,
        class CountGuard = std::enable_if_t<sizeof...(Ints) < TRank>,
        class TypeGuard = std::enable_if_t<(... && std::is_integral_v<Ints>)>>
    __device__ inline DeviceArray<scalar_t, TRank - sizeof...(Ints)> operator()(
        Ints const... indices) const {
        constexpr int out_rank = TRank - sizeof...(Ints);
        assert(out_rank > 0);
        DeviceArray<scalar_t, out_rank> out;

        // Offset data pointer.
        int offset = 0;
        int i = 0;
        for (const auto index : {indices...}) {
            assert(index >= 0);
            assert(index < shape[i]);
            offset += index * strides[i];
            i++;
        }
        out.data = data + offset;

        // Copy shape and strides.
        for (i = 0; i < out_rank; i++) {
            out.shape[i] = shape[sizeof...(Ints) + i];
            out.strides[i] = strides[sizeof...(Ints) + i];
        }
        return out;
    }

    // When the index count matches the rank of our array, we return a scalar
    // reference.
    template <
        typename... Ints,
        class CountGuard = std::enable_if_t<sizeof...(Ints) == TRank>,
        class TypeGuard = std::enable_if_t<(... && std::is_integral_v<Ints>)>>
    __device__ inline scalar_t &operator()(Ints const... indices) const {
        // Offset data pointer.
        int offset = 0;
        int i = 0;
        for (const auto index : {indices...}) {
            assert(index >= 0);
            assert(index < shape[i]);
            offset += index * strides[i];
            i++;
        }
        return data[offset];
    }

    // Check that the array matches an expected shape.
    template <typename... Ints>
    __device__ inline DeviceArray<scalar_t, TRank> &assert_shape(
        Ints const &... dims) {
        static_assert(sizeof...(Ints) == TRank, "Wrong dimension count.");
        int i = 0;
        for (const auto dim : {dims...}) {
            assert(dim == shape[i++]);
        }
        return *this;
    }

    scalar_t *data;  // Should be a device pointer.
    int shape[TRank];
    int strides[TRank];
};

// From: https://github.com/dfm/extending-jax
inline void ThrowIfError(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
}  // namespace jax_utils
