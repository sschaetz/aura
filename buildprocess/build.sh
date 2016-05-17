#!/bin/bash

set -ev

function build_and_test_target() 
{
        CURDIR=$(pwd)
        BASEDIR=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
        echo $BASEDIR
        rm -rf /tmp/aura-build
        mkdir -p /tmp/aura-build
        cd /tmp/aura-build
        cmake $BASEDIR/../ -DAURA_BASE=$1
        make all -s -j8
        ctest
        cd $CURDIR
}

build_and_test_target "CUDA"
build_and_test_target "METAL"
build_and_test_target "OPENCL"

