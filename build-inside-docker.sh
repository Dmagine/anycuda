#!/bin/bash

# Depends on build.sh

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

apt update && apt install -y --no-install-recommends curl
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
apt install -y --no-install-recommends cmake libvdpau-dev

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export C_INCLUDE_PATH=${CUDA_HOME}/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}

bash build.sh
