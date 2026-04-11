import glob
import os

import torch  # noqa: F401
import torch_npu  # noqa: F401
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


USE_NINJA = os.getenv("USE_NINJA") == "1"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
custom_op_name = os.getenv("CUSTOM_OP_NAME", "custom_ops_lib")
custom_op_version = os.getenv("CUSTOM_OP_VERSION", "0.0")

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)

ext = NpuExtension(
    name=custom_op_name,
    sources=source_files,
    extra_compile_args=[
        "-I" + os.path.join(os.path.dirname(os.path.abspath(torch_npu.__file__)), "include/third_party/acl/inc"),
    ],
)

setup(
    name=f"llm4ascendc-{custom_op_name}",
    version=str(custom_op_version),
    keywords="llm4ascendc custom_ops",
    ext_modules=[ext],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)

