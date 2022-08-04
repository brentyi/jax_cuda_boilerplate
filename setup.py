# Modified from:
# > https://github.com/wjakob/nanobind_example

import pathlib
import sys

try:
    import nanobind
    from skbuild import setup
except ImportError:
    print(
        "The preferred way to invoke 'setup.py' is via pip, as in 'pip "
        "install .'. If you wish to run the setup script directly, you must "
        "first install the build dependencies listed in pyproject.toml!",
        file=sys.stderr,
    )
    raise

package_name = "jax_cuda_boilerplate"

setup(
    name=package_name,
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    url="https://github.com/brentyi/jax_cuda_boilerplate",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    license="BSD",
    packages=[package_name],
    package_dir={"": "src"},
    cmake_install_dir="src/jax_cuda_boilerplate",
    include_package_data=True,
    package_data={package_name: ["py.typed", "**/*.pyi"]},
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.3.20",
        "jaxlib",
        "dcargs",
    ],
)
