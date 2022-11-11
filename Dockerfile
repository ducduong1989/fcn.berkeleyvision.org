FROM venkai/nvcaffe:17.0-cuda10.1-cudnn7-devel-ubuntu16.04

ARG TZ="Asia/Tokyo"

ENV TERM=xterm \
    LANG='C.UTF-8'  \
    LC_ALL='C.UTF-8' \
    TZ=${TZ}
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq ca-certificates libglib2.0-0 poppler-utils supervisor libgl1 tzdata git libxrender1 ffmpeg libsm6 nano libxext6 xcb  curl && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /input_medical_data
RUN mkdir /output_medical_data
RUN mkdir /output_segmentor
RUN mkdir /pretrained_model

# COPY . /workspace
WORKDIR /workspace/voc-fcn-alexnet

RUN echo 'export PYTHONPATH=$PYTHONPATH:/workspace' >> /root/.bashrc

CMD ["/bin/bash"]