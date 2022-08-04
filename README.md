# jax_cuda_boilerplate

Toy package containing boilerplate for combining custom CUDA kernels with JAX.
Tested on an 2080 TI; experimental but may be useful as a reference.

Similar to [dfm/extending-jax](https://github.com/dfm/extending-jax), but:

- Replaces deprecated CustomCall APIs with the latest MHLO equivalents.
- Replaces `pybind11` with `nanobind`, which is lighter and faster.
- Uses `skbuild` to (significantly) simplify the `setup.py` script.
- Implements a high-level API for indexing into n-dimensional GPU arrays.
- Includs utilities for passing shape information from Python to C++. This is
  useful for both stride-based indexing and determining block and grid
  dimensions for kernels.

### Install

```
# Core.
git clone git@github.com:brentyi/jax_cuda_boilerplate.git
cd jax_cuda_boilerplate
pip install -e .

# Generate stubs, formatting, etc.
pip install mypy isort black
stubgen -o src -p jax_cuda_boilerplate
isort --profile black . && black --preview . && black .
```

### Run

We can compare our toy CUDA kernel against a native-JAX implementation.

```
$ python scripts/compare.py --help
usage: compare.py [-h] [--num-rays-values INT [INT ...]] [--num-samples-values INT [INT ...]]

Compare our toy ray sampling implementation against a native JAX one. (the native
JAX one will usually be faster!)

╭─ arguments ──────────────────────────────────────────────────────────────────────────╮
│ -h, --help        show this help message and exit                                    │
│ --num-rays-values INT [INT ...]                                                      │
│                   Number of rays to batch for each run. (default: 8192)              │
│ --num-samples-values INT [INT ...]                                                   │
│                   Number of samples to generate for each ray. (default: 128 256 512) │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

This will assert that values match, then print timings (lower is better!):

```
$ python scripts/compare.py --num-rays-values 8192 16384 --num-samples-values 256 1024
num_rays=8192
        num_samples=256
                Just JAX (micros):       101.33
                Our CUDA (micros):       100.14 -1.2%

        num_samples=1024
                Just JAX (micros):       234.60
                Our CUDA (micros):       262.02 +11.7%

num_rays=16384
        num_samples=256
                Just JAX (micros):       148.06
                Our CUDA (micros):       145.20 -1.9%

        num_samples=1024
                Just JAX (micros):       452.52
                Our CUDA (micros):       556.95 +23.1%
```
