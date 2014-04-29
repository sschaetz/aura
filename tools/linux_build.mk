# makefile to build both cuda and opencl versions of aura on linux
#
# run something like
# 	make -j2 -f ./tools/linux_build.mk cuda opencl
# to build cuda and opencl versions in parallel

THIS_MAKEFILE_PATH:=$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
THIS_DIR:=$(shell cd $(dir $(THIS_MAKEFILE_PATH));pwd)

cmake_cuda: 
	mkdir -p /tmp/aura-cuda-build
	cd /tmp/aura-cuda-build; \
	cmake -DAURA_BACKEND=CUDA $(THIS_DIR)/../

cuda: cmake_cuda 
	cd /tmp/aura-cuda-build; \
	$(MAKE) 

cmake_opencl: 
	mkdir -p /tmp/aura-opencl-build
	cd /tmp/aura-opencl-build; \
	cmake -DAURA_BACKEND=OPENCL $(THIS_DIR)/../

opencl: cmake_opencl 
	cd /tmp/aura-opencl-build; \
	$(MAKE) 

clean:
	rm -rf /tmp/aura-opencl-build
	rm -rf /tmp/aura-cuda-build

all: cuda opencl

test: all
	cd /tmp/aura-cuda-build
	$(MAKE) test	
	cd /tmp/aura-opencl-build
	$(MAKE) test

