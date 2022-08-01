// Pybind logic for interfacing between Python and C++.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels.h"
#include "utils.h"

namespace cuda_ray_jax {

namespace {

namespace py = pybind11;

py::dict CustomCallTargets() {
    py::dict dict;
    dict["cuda_raysample_f32"] =
        py::capsule(reinterpret_cast<void*>(&cuda_raysample_f32),
                    "xla._CUSTOM_CALL_TARGET");
    return dict;
}

PYBIND11_MODULE(ops, m) {
    m.def("custom_call_targets", &CustomCallTargets);

    // Register TensorShapes helper.
    py::class_<utils::ShapesDescriptor>(m, "ShapesDescriptor")
        .def(py::init(&utils::ShapesDescriptor::construct))
        .def("as_bytes", [](const utils::ShapesDescriptor& shapes) {
            return py::bytes(reinterpret_cast<const char*>(&shapes),
                             sizeof(utils::ShapesDescriptor));
        });
}

}  // namespace

}  // namespace cuda_ray_jax
