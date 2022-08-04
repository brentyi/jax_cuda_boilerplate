// Bindings for utilities targeted at writing custom CUDA kernels for JAX.

#include <nanobind/nanobind.h>

#include <iostream>

#include "jax_utils.h"
#include "kernels.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(jax_utils_ext, m) {
    // Register ShapesDescriptor helper.
    nb::class_<jax_utils::ShapesDescriptor>(m, "ShapesDescriptor")
        .def_static(
            "construct",
            [](nb::list shapes) {
                assert(shapes.size() > 0);

                jax_utils::ShapesDescriptor out;
                out.count = shapes.size();
                out.ranks_cumsum[0] = 0;
                for (int i = 0; i < out.count; i++) {
                    auto shape = nb::cast<nb::tuple>(shapes[i]);
                    for (int j = 0; j < shape.size(); j++) {
                        out.flattened_dimensions[out.ranks_cumsum[i] + j] =
                            nb::cast<int>(shape[j]);
                    }
                    out.ranks[i] = shape.size();
                    if (i != out.count - 1) {
                        out.ranks_cumsum[i + 1] =
                            out.ranks_cumsum[i] + shape.size();
                        assert(out.ranks_cumsum[i] < jax_utils::MAX_DIMENSIONS);
                    }
                }
                return out;
            },
            "shapes"_a,
            "Instantiate a shape descriptor. Expects a list of tuples of "
            "integers.")
        .def(
            "as_bytes",
            [](const jax_utils::ShapesDescriptor& shapes) {
                return nb::bytes(reinterpret_cast<const char*>(&shapes),
                                 sizeof(jax_utils::ShapesDescriptor));
            },
            "Naive serialization for a shape descriptor.");
}

}  // namespace
