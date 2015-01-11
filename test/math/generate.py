# generate data for math tests, in particular the FFT

import os
import numpy as np

# 1d 4 samples
signal_1d_4 = np.array([1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j])
spectrum_1d_4 = np.fft.fft(signal_1d_4)

# 2d 4x4 samples
signal_2d_4 = np.tile(signal_1d_4, [4,1])
spectrum_2d_4 = np.fft.fft2(signal_2d_4)

# 3d 4x4x4 samples
signal_3d_4 = np.tile(signal_1d_4, [4,4,1])
spectrum_3d_4 = np.fft.fftn(signal_3d_4)

def cprint(name, A):
    print("const cfloat ", name, "[] = {", sep="", end="\n\t")
    el = 0
    for x in np.nditer(A, order='C'):
        if el%4 == 0:
            en = "\n\t"
        else:
            en = ""
        if el > 0:
            print(', ', end=en)
        print('cfloat(', np.real(x), ', ', np.imag(x), ')', end="", sep="")
        el+=1
    print("\n};", sep="")

print("#include <complex>", os.linesep)
print("typedef std::complex<float> cfloat;", os.linesep)

cprint('signal_1d_4', signal_1d_4)
print("")
cprint('spectrum_1d_4', spectrum_1d_4)
print(os.linesep)

cprint('signal_2d_4', signal_2d_4)
print("")
cprint('spectrum_2d_4', spectrum_2d_4)
print(os.linesep)

cprint('signal_3d_4', signal_3d_4)
print("")
cprint('spectrum_3d_4', spectrum_3d_4)
print(os.linesep)

