"""
Implemented by Facebook, Inc. and its affiliates.
Edited by Mustafa B. Yaldiz
"""
import torch
import detectron2


_DEEPFORMABLE_ENV_SETUP_DONE = False

def setup_environment():
    # Perform environment setup work.
    global _DEEPFORMABLE_ENV_SETUP_DONE
    if _DEEPFORMABLE_ENV_SETUP_DONE:
        return
    _DEEPFORMABLE_ENV_SETUP_DONE = True

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))
    
    # fmt: off
    assert get_version(torch) >= (1, 6), "Requires torch>=1.6"
    assert get_version(detectron2, digit=3) >= (0, 4, 1), "Requires detectron2>=0.4.1"
    # assert get_version(shapely) >= (1, 7, 1), "Requires shapely>=1.7.1"
    # import shapely
    # fmt: on


