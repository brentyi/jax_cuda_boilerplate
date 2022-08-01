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
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace utils {

constexpr uint32_t MAX_TENSORS = 8;
constexpr uint32_t MAX_DIMENSIONS = 16;

using ShapeAndRank = std::pair<const int32_t *, int32_t>;

// Simple descriptor for passing array shapes to our CUDA kernels. This is
// designed to be brutally simple to serialize and pass as the `opaque`
// parameter of our XLA custom calls. We might consider including more general
// metadata in the future, but just having all of the buffer shapes should (I
// think) cover most of the corner cases.
//
// As per the XLA documentation, we use `opaque` instead of putting shape
// information in a buffer because the shape information is often needed on the
// host to determine block and grid dimensions.
//
// Meant to live entirely in host memory and be used to construct NDArray
// objects, which are then copied to device memory.
class ShapesDescriptor {
   public:
    // Meant to be called via pybind.
    static ShapesDescriptor construct(
        const std::vector<std::vector<int32_t>> &shapes) {
        assert(shapes.size() < MAX_TENSORS);

        ShapesDescriptor out;
        out.count_ = 0;
        out.ranks_cumsum_[0] = 0;
        for (int32_t &i = out.count_; i < shapes.size(); i++) {
            const auto &shape = shapes[i];
            out.ranks_[i] = shape.size();

            memcpy(out.flattened_dimensions_ + out.ranks_cumsum_[i],
                   shape.data(), shape.size() * sizeof(shape[0]));
            out.ranks_cumsum_[i + 1] = out.ranks_cumsum_[i] + shape.size();

            if (i != shapes.size() - 1)
                assert(out.ranks_cumsum_[i] < MAX_DIMENSIONS);
        }
        return out;
    }

    // Deserialization; this is just a pointer cast with some very basic error
    // checking.
    static const ShapesDescriptor *from_bytes(const char *bytes) {
        auto *out = reinterpret_cast<const ShapesDescriptor *>(bytes);
        assert(out->canary_ == 94709);
        return out;
    }

    ShapeAndRank get(int32_t i) const {
        return std::make_pair(flattened_dimensions_ + ranks_cumsum_[i],
                              ranks_[i]);
    }

   private:
    int32_t count_;
    int32_t ranks_[MAX_TENSORS];
    int32_t ranks_cumsum_[MAX_TENSORS];
    int32_t flattened_dimensions_[MAX_DIMENSIONS];
    int32_t canary_ = 94709;
};

// Structure for making the NDArray constructor more succinct.
struct ArrayBundle {
    void **buffers;
    const ShapesDescriptor *shapes;
};

// Light helper for working with+indexing into dense N-dimensional arrays.
// Mirroring JAX, we assume everything is row-major.
//
// Designed to be instantiated on the host, then passed to the device.
template <typename Scalar, int32_t TRank>
class NDArray {
   public:
    // Standard constructor; computes shape and stride.
    NDArray(Scalar *data, ShapeAndRank shape_and_rank) : data(data) {
        assert(shape_and_rank.second == TRank);

        int32_t cumprod = 1;
        for (int32_t i = 0; i < TRank; i++) {
            this->shape[i] = static_cast<int32_t>(shape_and_rank.first[i]);
            this->strides[TRank - 1 - i] = cumprod;
            cumprod *= shape_and_rank.first[TRank - 1 - i];
        }
    }

    // Initialization from an ArrayBundle.
    NDArray(const ArrayBundle &arrays, int32_t i)
        : NDArray(reinterpret_cast<Scalar *>(arrays.buffers[i]),
                  arrays.shapes->get(i)) {}

    // Eigen-style indexing syntax, eg `array3d(i, j, k)`.
    template <typename... Ints>
    __device__ inline Scalar &operator()(Ints const &... indices) const {
        static_assert(sizeof...(Ints) == TRank,
                      "Expected one index per dimension.");
        uint32_t flat_index = 0, i = 0;
        (..., (flat_index += indices * strides[i++]));
        return data[flat_index];
    }

    Scalar *data;  // Location in GPU memory.
    int32_t shape[TRank];
    int32_t strides[TRank];
};

// From: https://github.com/dfm/extending-jax
inline void ThrowIfError(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
}  // namespace utils
