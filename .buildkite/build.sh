#!/usr/bin/env bash

set -ueo pipefail

mkdir -p build && cd build
cmake -DAURA_BASE="$AURA_BASE" -DAURA_UNIT_TEST_DEVICE="$AURA_UNIT_TEST_DEVICE" ..
make all
ctest
