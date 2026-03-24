"""safe_ops C++/CUDA 확장 빌드 스크립트"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="safe_ops",
    ext_modules=[
        CUDAExtension(
            name="safe_ops",
            sources=["safe_ops.cpp", "safe_ops_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-march=native"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
