"""Builds the optional CUDA extension when nvcc is available; otherwise installs
the pure-Python/Triton package (the Triton kernel is the default GPU path)."""
import os

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}
try:
    import torch  # noqa: F401  (ensures torch is importable before cpp_extension)
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if CUDA_HOME is not None and os.environ.get("FUSED_ATTN_SKIP_CUDA", "0") != "1":
        arch_flags = os.environ.get("TORCH_CUDA_ARCH_LIST")  # e.g. "8.0;9.0"
        nvcc = ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"]
        ext_modules.append(
            CUDAExtension(
                name="fused_attn._C",
                sources=["csrc/bindings.cpp", "csrc/flash_attn.cu"],
                include_dirs=[os.path.abspath("csrc")],
                extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc},
            )
        )
        cmdclass["build_ext"] = BuildExtension
except ImportError:
    pass

setup(
    name="fused-attn",
    version="0.2.0",
    packages=find_packages(exclude=("tests", "benchmarks")),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=["torch>=2.4", "triton>=3.0"],
    extras_require={"dev": ["pytest", "matplotlib"], "bench": ["flash-attn"]},
)
