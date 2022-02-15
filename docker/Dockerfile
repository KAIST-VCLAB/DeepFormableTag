FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# If Cuda version is never than your host, you must modify above image with 
# the supported one and cudatoolkit=$cuda_version below for pytorch installation.

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates git wget vim cmake ninja-build build-essential curl \
    	libjpeg-dev libpng-dev x11-apps v4l-utils unzip \
		rsync ffmpeg psmisc libcairo2-dev libgif-dev libpango1.0-dev \
  	&& rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && chmod +x ~/miniconda.sh \
     && ~/miniconda.sh -b -p /opt/conda \
     && rm ~/miniconda.sh \
     && /opt/conda/bin/conda install -y python=3.7 \
	 && /opt/conda/bin/conda install -y pytorch=1.8.2 torchvision=0.9.2 cudatoolkit=11.1 -c pytorch-lts -c nvidia \
	 &&  /opt/conda/bin/conda install -y tensorboard pycairo matplotlib scikit-image jupyter ninja cython \
	 		typing future pytest scipy scikit-learn \
	 && /opt/conda/bin/conda install -y -c conda-forge plyfile \
	 && /opt/conda/bin/python -m pip install -U  pycocotools==2.0.4 \
	 		shapely==1.8.0 opencv-contrib-python==4.5.5.62 kornia==0.6.2 \
			glfw ipympl pyrr future-fstrings PyOpenGL PyOpenGL_accelerate \
	 && /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
ENV PATH /root/.local/bin:$PATH
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV FVCORE_CACHE="/tmp"

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo \
     && cd detectron2_repo && git checkout "v0.6" \
     && python -m pip install -e .

# Installation command for original AprilTag implementation
# RUN git clone --recurse-submodules https://github.com/AprilRobotics/apriltag.git \
# 	&& cd apriltag && mkdir /root/.local && ln -s /opt/conda/lib /root/.local/lib \
#     && mkdir build && cd build \
# 	&& cmake -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_PACKAGES_PATH=/opt/conda/lib/python3.7/site-packages/ \
#      -DCMAKE_INSTALL_PREFIX=/opt/conda/ -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
#      -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") ..\
#     && make -j8 && make install

WORKDIR /