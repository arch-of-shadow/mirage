# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git ninja-build curl wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mv Miniconda3-latest-Linux-x86_64.sh ~/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/Miniconda3-latest-Linux-x86_64.sh && \
    ~/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-latest-Linux-x86_64.sh

RUN /opt/conda/bin/conda remove -y --override-channels -c conda-forge conda-anaconda-tos && \
    /opt/conda/bin/conda install -y --override-channels -c conda-forge \
    python=3.11 \
    conda-build \
    conda-verify \
    cmake \
    make \
    numpy \
    cython && \
    /opt/conda/bin/conda clean -ya
    
ENV PATH=/opt/conda/bin:$PATH
ENV MIRAGE_HOME=/mirage/
ENV LD_LIBRARY_PATH=/mirage/deps/z3/build/:$LD_LIBRARY_PATH

WORKDIR /mirage
#RUN chmod -R a+w /usr/mirage
