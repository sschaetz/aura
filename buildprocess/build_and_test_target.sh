function build_and_test_target() 
{
        BASEDIR=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
        mkdir -p /tmp/aura-build
        pushd /tmp/aura-build
        rm -rf ./*
        cmake $BASEDIR/../ -DAURA_BASE=$1 -DAURA_UNIT_TEST_DEVICE=$4 > /tmp/aura-build-$1.log 2>&1
        CMAKE_RESULT=$?
        make all -s -j8 >> /tmp/aura-build-$1.log 2>&1
        MAKE_RESULT=$?
        ctest >> /tmp/aura-build-$1.log 2>&1
        CTEST_RESULT=$?
        echo "cmake $CMAKE_RESULT make $MAKE_RESULT ctest $CTEST_RESULT"
        ./test/test.alang >> /tmp/aura-build-$1.log 2>&1
        popd
}
