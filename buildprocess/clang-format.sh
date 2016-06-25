#!/bin/bash

BASEDIR=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")

find $BASEDIR -iname *.hpp -o -iname *.cpp | xargs clang-format -i -style=file
