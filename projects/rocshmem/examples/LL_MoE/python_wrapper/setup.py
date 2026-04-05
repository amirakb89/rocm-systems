"""
Build the mori_v1ll_cpp torch extension for ROCm + rocSHMEM.

Usage (inside the rocshmem-amir container):
    ROCSHMEM_DIR=/root/rocshmem python setup.py install
or:
    pip install -e .
"""
import os
import setuptools

rocm_path = os.getenv("ROCM_HOME", "/opt/rocm")
rocshmem_dir = os.getenv("ROCSHMEM_DIR", os.path.expanduser("~/rocshmem"))
target_arch = os.getenv("TARGET_GPU_ARCH", "gfx942")

assert os.path.isdir(rocshmem_dir), f"ROCSHMEM_DIR not found: {rocshmem_dir}"

os.environ["PYTORCH_ROCM_ARCH"] = target_arch
os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] = "1"
os.environ["CC"] = f"{rocm_path}/bin/hipcc"
os.environ["CXX"] = f"{rocm_path}/bin/hipcc"

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_flags = [
    "-O3",
    "-DUSE_ROCM=1",
    "-fgpu-rdc",
    "-Wno-unused-variable",
    "-Wno-sign-compare",
]
nvcc_flags = ["-O3", "-DUSE_ROCM=1", "-fgpu-rdc"]

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

setuptools.setup(
    name="mori_v1ll",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="mori_v1ll_cpp",
            sources=[
                "csrc/v1ll_ops.cu",
                "csrc/v1ll_module.cu",
            ],
            include_dirs=[
                "csrc/",
                parent_dir,
                f"{rocshmem_dir}/include",
            ],
            library_dirs=[f"{rocshmem_dir}/lib"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            extra_link_args=[
                f"-l:librocshmem.a",
                f"-Wl,-rpath,{rocshmem_dir}/lib",
                "-fgpu-rdc",
                "--hip-link",
                "-lamdhip64",
                "-lhsa-runtime64",
                f"--offload-arch={target_arch}",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
