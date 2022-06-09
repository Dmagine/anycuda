#!/bin/bash

# Depends on build-inside-docker.sh

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

CUDA_IMAGE=${ANYCUDA_BUILD_CUDA_IMAGE:-nvidia/cuda:11.2.0-devel-ubuntu18.04}
docker run --rm --user="$(id -u):$(id -g)" -v "$PWD":/anycuda -w /anycuda $CUDA_IMAGE bash build-inside-docker.sh
