# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
from .build import MARKER_GENERATOR_REGISTRY, build_marker_generator
from .generalized_generator import GeneralizedGenerator, KDTreeClassPredictor
# from .aruco_generator import ArucoGenerator
# You need to install AprilTag, described in Dockerfile
# from .april_generator import AprilGenerator   
# from .image_generator import ImageGenerator
