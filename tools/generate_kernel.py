# this script generates proper include files from kernel codes
# it takes into consideration the backend

# call it with:
# pythong generate_kernel.py <absolute path to reco-repository> <mode>
# supported modes are: 
#  * clean: delete temporary files
#  * opencl: copy the contents of the file into a string in the header file
#  * cuda: run nvcc, generate PTX, copy result into a string in the header file

# list of kernel files
kernels = [
    "include/boost/aura/math/split_interleaved_i2s.cc",
    "include/boost/aura/math/split_interleaved_s2i.cc",
    "include/boost/aura/math/memset_zero_float.cc",
    ];

import sys

mode = 'cuda'

# repository base dir
if len(sys.argv) < 2:
    # error
    raise Exception('Please specify as first argument the path ' +
        'to the reco repository.')
else:
    bdir = sys.argv[1];
    bdir += '/'

# mode 
if len(sys.argv) > 2:
    mode = sys.argv[2]
    mode = mode.lower()


includedir = bdir + "/include/"
generateddir= bdir + "/generated/"

from shutil import copyfile
import os
from subprocess import call

def mkdirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

mkdirs(generateddir)

for k in kernels:
    k = bdir + k
    print (k)
    fname = os.path.splitext(os.path.basename(k))[0]
    tname = generateddir+fname+".cu"
    pname = generateddir+fname+".ptx"
    hname = generateddir+fname+"_kernels.hpp"
   
    if (mode == 'clean'):
        if os.path.exists(tname):
            os.remove(tname)
        if os.path.exists(pname):
            os.remove(pname)
        if os.path.exists(hname):
            os.remove(hname)
    elif (mode == 'cuda'):
        # generate PTX from each kernel file
        copyfile(k, tname)
        os.system("/usr/local/cuda/bin/nvcc -I "+includedir+
            " -DAURA_BACKEND_CUDA -o=" + pname + 
            " -arch=compute_20 --ptx " + tname)

        # create hpp files containing PTX kernel strings
        with open(hname, 'w+') as ofh:
            ofh.write('static const char* ' + fname + '_kernels = R"aura_kernel(\n')
            with open(pname) as ifh:
                lines = ifh.readlines()
                ofh.writelines(lines)
            ofh.write('\n)aura_kernel";');
    elif (mode == 'opencl'):
        # inject data from cc file into header file
        with open(hname, 'w+') as ofh:
            ofh.write('static const char* ' + fname + '_kernels = R"aura_kernel(\n')
            with open(k) as ifh:
                lines = ifh.readlines()
                ofh.writelines(lines)
            ofh.write('\n)aura_kernel";');
    else:
        raise Exception('Unknown mode, please select cuda, opencl or clean.')

