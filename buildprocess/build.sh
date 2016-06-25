#!/bin/bash

function build_and_test_target() 
{
        CURDIR=$(pwd)
        BASEDIR=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
        echo $BASEDIR
        rm -rf /tmp/aura-build
        mkdir -p /tmp/aura-build
        cd /tmp/aura-build
        cmake $BASEDIR/../ -DAURA_BASE=$1 > /tmp/aura-build-$1.log 2>&1
        CMAKE_RESULT=$?
        make all -s -j8 >> /tmp/aura-build-$1.log 2>&1
        MAKE_RESULT=$?
        ctest >> /tmp/aura-build-$1.log 2>&1
        CTEST_RESULT=$?
        cd $CURDIR
        echo "CMAKE" $CMAKE_RESULT "MAKE" $MAKE_RESULT CTEST $CTEST_RESULT
        if [ $# -ge 3 ]
        then
                if [ ! -z $4 ] 
                then 
                        curl --upload-file /tmp/aura-build-$1.log $4newbr?cmake_result=$CMAKE_RESULT\&make_result=$MAKE_RESULT\&ctest_result=$CTEST_RESULT\&branch=$2\&commitid=$3\&backend=$1\&machine=$(hostname)
                fi
        fi
        }

BRANCH=$(git branch | sed -n '/\* /s///p')
COMMIT=$(git rev-parse HEAD)

build_and_test_target "CUDA" $BRANCH $COMMIT $1
build_and_test_target "METAL" $BRANCH $COMMIT $1
build_and_test_target "OPENCL" $BRANCH $COMMIT $1
rm -rf /tmp/aura-build
