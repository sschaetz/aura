#!/usr/bin/env bash

set -ueo pipefail

brew install cmake
brew install boost

my_dir="$(dirname "$0")"
. "$my_dir/../buildprocess/build_and_test_target.sh"

echo "INCLUDE OK"

if [ -n "$BUILDKITE_AGENT_META_DATA_CUDA" ]
then
        if [ "$BUILDKITE_AGENT_META_DATA_CUDA" == "true" ]
        then
                echo "CUDA OK"
                mkdir -p aura-build-cuda
                cd aura-build-cuda
                cmake -DAURA_BASE=CUDA ../
                make -j8
                ctest
                cd ..
                rm -rf ./aura-build-cuda
        fi
fi

if [ -n "$BUILDKITE_AGENT_META_DATA_OPENCL" ]
then
        if [ "$BUILDKITE_AGENT_META_DATA_OPENCL" == "true" ]
        then
                echo "OPENCL OK"
                mkdir -p aura-build-opencl
                cd aura-build-opencl
                cmake -DAURA_BASE=OPENCL -DAURA_UNIT_TEST_DEVICE=1 ../
                make -j8
                ctest
                cd ..
                rm -rf ./aura-build-opencl
        fi
fi

if [ -n "$BUILDKITE_AGENT_META_DATA_METAL" ]
then
        if [ "$BUILDKITE_AGENT_META_DATA_METAL" == "true" ]
        then
                echo "METAL OK"
                mkdir -p aura-build-metal
                cd aura-build-metal
                cmake -DAURA_BASE=METAL ../
                make -j8
                ctest
                cd ..
                rm -rf ./aura-build-metal
        fi
fi
