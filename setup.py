#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import importlib.util
import os
import platform
import runpy
import shutil
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path
from typing import List, Optional

import torch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension

MOJO_VERSION = "1.0.0b2"


def _mojo_compiler() -> Optional[str]:
    native = sys.platform == "darwin" and platform.machine() == "arm64"
    mode = os.getenv("PYTORCH3D_BUILD_MOJO", "required" if native else "0").lower()
    if mode not in {"0", "auto", "required"}:
        raise ValueError("PYTORCH3D_BUILD_MOJO must be 0, auto, or required")
    compiler = None
    if native and mode != "0":
        environment_compiler = Path(sys.executable).with_name("mojo")
        compiler = (
            str(environment_compiler)
            if environment_compiler.is_file()
            else shutil.which("mojo")
        )
    if compiler is not None:
        version = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout
        if not version.startswith(f"Mojo {MOJO_VERSION} "):
            message = f"Mojo {MOJO_VERSION} is required; found {version.strip()}"
            if mode == "required":
                raise RuntimeError(message)
            warnings.warn(message)
            compiler = None
    if mode == "required" and compiler is None:
        raise RuntimeError("Mojo is required for this build but is unavailable")
    return compiler


MOJO_COMPILER = _mojo_compiler()


def get_existing_ccbin(nvcc_args: List[str]) -> Optional[str]:
    """
    Given a list of nvcc arguments, return the compiler if specified.

    Note from CUDA doc: Single value options and list options must have
    arguments, which must follow the name of the option itself by either
    one of more spaces or an equals character.
    """
    last_arg = None
    for arg in reversed(nvcc_args):
        if arg == "-ccbin":
            return last_arg
        if arg.startswith("-ccbin="):
            return arg[7:]
        last_arg = arg
    return None


def get_extensions():
    no_extension = os.getenv("PYTORCH3D_NO_EXTENSION", "0") == "1"
    if no_extension:
        msg = "SKIPPING EXTENSION BUILD. PYTORCH3D WILL NOT WORK!"
        print(msg, file=sys.stderr)
        warnings.warn(msg)
        return []

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "pytorch3d", "csrc")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
    extension = CppExtension

    extra_compile_args = {"cxx": ["-std=c++17"]}
    define_macros = []
    include_dirs = [extensions_dir]

    torch_version = tuple(
        int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
    )
    if sys.platform == "darwin" and torch_version < (2, 8):
        extra_compile_args["cxx"].append("-Wno-invalid-specialization")
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    force_no_cuda = os.getenv("PYTORCH3D_FORCE_NO_CUDA", "0") == "1"
    if (
        not force_no_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    ) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        # Thrust is only used for its tuple objects.
        # With CUDA 11.0 we can't use the cudatoolkit's version of cub.
        # We take the risk that CUB and Thrust are incompatible, because
        # we aren't using parts of Thrust which actually use CUB.
        define_macros += [("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        cub_home = os.environ.get("CUB_HOME", None)
        nvcc_args = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.name != "nt":
            nvcc_args.append("-std=c++17")

        # CUDA 13.0+ compatibility flags for pulsar.
        # Starting with CUDA 13, __global__ function visibility changed.
        # See: https://developer.nvidia.com/blog/
        #      cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            major = int(cuda_version.split(".")[0])
            if major >= 13:
                nvcc_args.extend(
                    [
                        "--device-entity-has-hidden-visibility=false",
                        "-static-global-template-stub=false",
                    ]
                )
        if cub_home is None:
            prefix = os.environ.get("CONDA_PREFIX", None)
            if prefix is not None and os.path.isdir(prefix + "/include/cub"):
                cub_home = prefix + "/include"

        if cub_home is None:
            warnings.warn(
                "The environment variable `CUB_HOME` was not found. "
                "NVIDIA CUB is required for compilation and can be downloaded "
                "from `https://github.com/NVIDIA/cub/releases`. You can unpack "
                "it to a location of your choice and set the environment variable "
                "`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
            )
        else:
            include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))
        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            nvcc_args.extend(nvcc_flags_env.split(" "))

        # This is needed for pytorch 1.6 and earlier. See e.g.
        # https://github.com/facebookresearch/pytorch3d/issues/436
        # It is harmless after https://github.com/pytorch/pytorch/pull/47404 .
        # But it can be problematic in torch 1.7.0 and 1.7.1
        if torch.__version__[:4] != "1.7.":
            CC = os.environ.get("CC", None)
            if CC is not None:
                existing_CC = get_existing_ccbin(nvcc_args)
                if existing_CC is None:
                    CC_arg = "-ccbin={}".format(CC)
                    nvcc_args.append(CC_arg)
                elif existing_CC != CC:
                    msg = f"Inconsistent ccbins: {CC} and {existing_CC}"
                    raise ValueError(msg)

        extra_compile_args["nvcc"] = nvcc_args

    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        extension(
            "pytorch3d._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Retrieve __version__ from the package.
__version__ = runpy.run_path("pytorch3d/__init__.py")["__version__"]


class BuildExtension(torch.utils.cpp_extension.BuildExtension):
    def __init__(self, *args, **kwargs):
        use_ninja = os.getenv("PYTORCH3D_NO_NINJA", "0") != "1"
        super().__init__(*args, use_ninja=use_ninja, **kwargs)

    def build_extensions(self):
        super().build_extensions()
        output = Path(self.get_ext_fullpath("pytorch3d._C")).parent / "_mojo.so"
        if MOJO_COMPILER is None:
            output.unlink(missing_ok=True)
            if self.inplace:
                (Path(__file__).parent / "pytorch3d/_mojo.so").unlink(missing_ok=True)
            return
        csrc_dir = Path(__file__).parent / "pytorch3d/csrc"
        self._mojo_output = output
        output.parent.mkdir(parents=True, exist_ok=True)
        deployment_target = os.getenv(
            "MACOSX_DEPLOYMENT_TARGET",
            sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET") or "11.0",
        )
        subprocess.run(
            [
                MOJO_COMPILER,
                "build",
                csrc_dir / "mojo/_mojo.mojo",
                "-I",
                csrc_dir / "point_mesh",
                "-I",
                csrc_dir / "rasterize_meshes",
                "--emit",
                "shared-lib",
                "--target-triple",
                f"arm64-apple-macosx{deployment_target}",
                "--target-cpu",
                "apple-m1",
                "-Xlinker",
                "-rpath",
                "-Xlinker",
                "@loader_path/../modular/lib",
                "-o",
                output,
            ],
            check=True,
            env={
                **os.environ,
                "MACOSX_DEPLOYMENT_TARGET": deployment_target,
                "MOJO_PYTHON": sys.executable,
            },
        )
        modular = importlib.util.find_spec("modular")
        runtime = Path(next(iter(modular.submodule_search_locations))) / "lib"
        self._mojo_runtime = runtime
        subprocess.run(
            [
                "xcrun",
                "install_name_tool",
                "-id",
                "@rpath/_mojo.so",
                "-delete_rpath",
                runtime,
                output,
            ],
            check=True,
        )

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        target = Path(__file__).parent / "pytorch3d/_mojo.so"
        if MOJO_COMPILER is None:
            target.unlink(missing_ok=True)
            return
        if hasattr(self, "_mojo_output"):
            shutil.copy2(self._mojo_output, target)
            subprocess.run(
                [
                    "xcrun",
                    "install_name_tool",
                    "-add_rpath",
                    self._mojo_runtime,
                    target,
                ],
                check=True,
            )


class BuildPy(build_py):
    def run(self):
        super().run()
        for cache in Path(self.build_lib).rglob("__pycache__"):
            shutil.rmtree(cache)


trainer = "pytorch3d.implicitron_trainer"

setup(
    name="pytorch3d",
    version=__version__,
    author="FAIR",
    url="https://github.com/facebookresearch/pytorch3d",
    description="PyTorch3D is FAIR's library of reusable components "
    "for deep Learning with 3D data.",
    packages=find_packages(
        exclude=("configs", "tests", "tests.*", "docs.*", "projects.*")
    )
    + [trainer],
    package_dir={trainer: "projects/implicitron_trainer"},
    install_requires=[
        "iopath",
        f'mojo=={MOJO_VERSION}; platform_system == "Darwin" and platform_machine == "arm64"',
    ],
    extras_require={
        "all": ["matplotlib", "tqdm>4.29.0", "imageio", "ipywidgets"],
        "dev": ["flake8", "usort"],
        "implicitron": [
            "hydra-core>=1.1",
            "visdom",
            "lpips",
            "tqdm>4.29.0",
            "matplotlib",
            "accelerate",
            "sqlalchemy>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            f"pytorch3d_implicitron_runner={trainer}.experiment:experiment",
            f"pytorch3d_implicitron_visualizer={trainer}.visualize_reconstruction:main",
        ]
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension, "build_py": BuildPy},
    package_data={
        "": ["*.json"],
    },
)
