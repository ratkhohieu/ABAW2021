# Build docker fisrt
# docker build . -t svenclary/pytorch_cv:lastest
# run make file ...
# Name image: svenclary/pytorch_cv:lastest
# sudo chmod 7777 -R KERC/

FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libturbojpeg \
    software-properties-common \
    git \
    ffmpeg\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN pip install albumentations
RUN pip install timm
RUN pip install pandas
RUN pip install matplotlib
RUN pip install -U scikit-learn
RUN pip install optuna
RUN pip install tqdm
RUN pip install facenet-pytorch
RUN pip install opencv-python
RUN conda install -c conda-forge libsndfile
RUN pip install librosa
RUN pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
RUN pip install face-alignment
RUN pip install tensorboardX
RUN pip install torchtext==0.7


#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace