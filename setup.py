"""
!/usr/bin/env python
Copyright (c) Facebook, Inc. and its affiliates.
Edited by Mustafa B. Yaldiz (VCLAB, KAIST)
"""
import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
# from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 6], "Requires PyTorch >= 1.6"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "deepformable", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("D2_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version

# PROJECTS = {}

setup(
    name="deepformable",
    version=get_version(),
    author="Mustafa B. YALDIZ",
    url="https://github.com/KAIST-VCLAB/DeepFormableTag",
    description="DeepformableTag is data-driven fiducial marker system.",
    packages=find_packages(),
    python_requires=">=3.7",
    # install_requires=[
    #     "detectron2>=0.4.1",
    #     "shapely>=1.7.1",
    # ],
)