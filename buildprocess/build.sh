#!/bin/bash

my_dir="$(dirname "$0")"

. "$my_dir/../buildprocess/build_and_test_target.sh"

BRANCH=$(git branch | sed -n '/\* /s///p')
COMMIT=$(git rev-parse HEAD)

build_and_test_target "CUDA" $BRANCH $COMMIT $1
build_and_test_target "METAL" $BRANCH $COMMIT $1
build_and_test_target "OPENCL" $BRANCH $COMMIT $1
rm -rf /tmp/aura-build
