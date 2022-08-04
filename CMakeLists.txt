cmake_minimum_required(VERSION 3.18...3.22)
project(jax_cuda_boilerplate LANGUAGES CXX CUDA)

if (NOT SKBUILD)
  message(WARNING "This CMake file should be executed via scikit-build. "
                  "Please run\n$ pip install .")
  set(PYTHON_EXECUTABLE "python")
else()
  # Constrain FindPython to find the Python version used by scikit-build.
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
endif()

# Generate `compile_commands.json`. This is useful for IDE support, clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# We like debug mode for assert statements!
set(CMAKE_BUILD_TYPE Debug)

# Configure CUDA.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")

# Find pybind.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Run `nanobind.cmake_dir()` from Python to detect install location
execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)

# Override optimization flag used for nanobind: note that the default, -Os,
# isn't supported by nvcc.
function(nanobind_opt_size name)
  if (MSVC)
    # set(NB_OPT_SIZE /O2)
    set(NB_OPT_SIZE /Os)
  else()
    # set(NB_OPT_SIZE -O2)
    set(NB_OPT_SIZE -O2)
  endif()

  target_compile_options(${name} PRIVATE
      $<$<CONFIG:Release>:${NB_OPT_SIZE}>
      $<$<CONFIG:MinSizeRel>:${NB_OPT_SIZE}>
      $<$<CONFIG:RelWithDebInfo>:${NB_OPT_SIZE}>)
endfunction()

# Our project!
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

nanobind_add_module(
  custom_calls_ext
  # Build static libnanobind (the extension module itself remains a shared library)
  NB_STATIC
  src/bindings_custom_calls.cc
  src/kernels.cu)

nanobind_add_module(
  jax_utils_ext
  # Build static libnanobind (the extension module itself remains a shared library)
  NB_STATIC
  src/bindings_jax_utils.cc)

install(TARGETS custom_calls_ext jax_utils_ext LIBRARY DESTINATION .)
set_target_properties(custom_calls_ext PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
set_target_properties(jax_utils_ext PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)

