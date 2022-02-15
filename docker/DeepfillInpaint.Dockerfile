FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-opencv wget git cmake \
    libcairo2-dev libjpeg-dev libpango1.0-dev \
    libgif-dev build-essential

RUN python -m pip install git+https://github.com/JiahuiYu/neuralgym \
    && python -m pip uninstall -y enum34 \
    && python -m pip install pyyaml opencv-python \
        opencv-contrib-python tqdm Pillow pycairo shapely
