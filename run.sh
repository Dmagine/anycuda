#!/usr/bin/bash
ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
echo "ROOT=$ROOT"
export LD_PRELOAD=$ROOT/build/libcuda-control.so
export LOGGER_LEVEL=4
./build/example
