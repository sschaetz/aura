#!/bin/bash

function build_and_test_target 
{
        BASEDIR=$(dirname "$0")
        mkdir -p /tmp/aura-build
        cd /tmp/aura-build
        cmake $BASEDIR/../ -DAURA_BASE=$1
        make all -s -j8
        ctest
}

build_and_test_target "METAL"
build_and_test_target "CUDA"
build_and_test_target "OPENCL"

