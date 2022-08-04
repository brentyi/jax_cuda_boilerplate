// Bindings for custom call targets. Exposes a `custom_call_targets` dictionary.

#include <nanobind/nanobind.h>

#include <iostream>

#include "kernels.h"

namespace {

namespace nb = nanobind;

NB_MODULE(custom_calls_ext, m) {
    m.def("custom_call_targets", []() {
        nb::dict dict;
        dict["cuda_raysample_f32"] = nb::capsule(
            reinterpret_cast<void *>(&jax_cuda_boilerplate::cuda_raysample_f32),
            "xla._CUSTOM_CALL_TARGET");
        return dict;
    });
}

}  // namespace
