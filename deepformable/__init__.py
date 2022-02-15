# Copyright (c) Facebook, Inc. and its affiliates.
# Edited by Mustafa B. Yaldiz (VCLAB, KAIST)

from .utils.env import setup_environment

setup_environment()


# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.1.0"