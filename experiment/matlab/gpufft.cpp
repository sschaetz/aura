#include "nullptr.hpp"

#include <aura/backend.hpp>
#include <aura/misc/profile.hpp>
#include "mex.h"
#include "matrix.h"

/**
 * 0) generate the data
 * 1) get data from matlab
 * 2) push it to GPU
 * 3) transpose and interleave it
 * 4) fft
 * 5) undo transpose and interleaving
 * 6) pull it from GPU
 * 7) return it to Matlab
 */

using namespace aura::backend;
using namespace aura;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double t1, t2;	

	// check for proper number of arguments
	if(nrhs!=1) {
	    mexErrMsgTxt("only one input required");
	}
	if(nlhs!=1) {
	    mexErrMsgTxt("only one output required");
	}

	// check for single values (we can't do double FFT on our GPUs)
	if(!mxIsSingle(prhs[0])) {
	    mexErrMsgTxt("double is not supported");
	}
	
	// check for complex values (we don't support real FFTs)
	if(!mxIsComplex(prhs[0])) {
	    mexErrMsgTxt("real is not supported");
	}

	// get dimensionality of argument 
	mwSize numdims = mxGetNumberOfDimensions(prhs[0]);
	if(2 != numdims && 3 != numdims) {
	    mexErrMsgTxt("only 2D and batched 2D transform supported");
	}

	// xdim, ydim and batchsize 
	const mwSize * dims = mxGetDimensions(prhs[0]);
	std::size_t x = dims[0];
	std::size_t y = dims[1];
	std::size_t b = (numdims > 2) ? dims[2] : 1;
	std::size_t size = x*y*b;


	// output	
	plhs[0] = mxCreateNumericArray(numdims, dims, 
		mxSINGLE_CLASS, mxCOMPLEX);
	float * result_r = reinterpret_cast<float*>(mxGetData(plhs[0]));
	float * result_i = reinterpret_cast<float*>(mxGetImagData(plhs[0]));
	cuMemHostRegister(result_r, x*y*b*sizeof(float),
		CU_MEMHOSTREGISTER_PORTABLE);	
	cuMemHostRegister(result_i, x*y*b*sizeof(float),
		CU_MEMHOSTREGISTER_PORTABLE);	
	
	// init gpu
	t1 = now();	
	device d(0);
	feed f(d);
	wait_for(f);
	t2 = now() - t1;
	printf("init gpu %f\n", t2/1e6);
	
	
	printf("free mem start %lu MB\n", device_get_free_memory(d));
	
	// allocate memory on gpu
	fft fh(d, f, bounds(x, y), fft::type::c2c, b);

	device_ptr<float> buffr = device_malloc<float>(x*y*b, d);
	device_ptr<float> buffi = device_malloc<float>(x*y*b, d);
	device_ptr<float> buffc = device_malloc<float>(2*x*y*b, d);
	
	// copy to gpu
	t1 = now();	
	copy(buffr, reinterpret_cast<float *>(mxGetData(prhs[0])), x*y*b, f);
	copy(buffi, reinterpret_cast<float *>(mxGetImagData(prhs[0])), x*y*b, f);
	wait_for(f);
	t2 = now() - t1;
	printf("copy cpu->gpu %f\n", t2/1e6);

	// complex to interleaved
	t1 = now();	
	module m =  create_module_from_file("kernel.ptx", d);
	kernel c2i = create_kernel(m, "kern_c2i");
	kernel i2c = create_kernel(m, "kern_i2c");
	wait_for(f);
	t2 = now() - t1;
	printf("compile kernel %f\n", t2/1e6);
	
	t1 = now();	
	invoke(c2i, mesh(x, y, b), 
		bundle(x), args(size, buffc.get(), buffr.get(), buffi.get()), f);
	wait_for(f);
	t2 = now() - t1;
	printf("c2i %f\n", t2/1e6);
	
	t1 = now();	
	fft_forward(buffc, buffc, fh, f);
	fft_inverse(buffc, buffc, fh, f);
	wait_for(f);
	t2 = now() - t1;
	printf("fft %f\n", t2/1e6);

	t1 = now();	
	invoke(i2c, mesh(x, y, b), 
		bundle(x), args(size, buffr.get(), buffi.get(), buffc.get()), f);
	wait_for(f);
	t2 = now() - t1;
	printf("i2c %f\n", t2/1e6);

	// copy from gpu and return
	t1 = now();	
	copy(result_r, buffr, x*y*b, f);
	copy(result_i, buffi, x*y*b, f);
	wait_for(f);
	t2 = now() - t1;
	printf("copy gpu->cpu %f\n", t2/1e6);
	
	device_free(buffr);
	device_free(buffi);
	device_free(buffc);
	cuMemHostUnregister(result_r);
	cuMemHostUnregister(result_i);
	
	printf("free mem stop %lu MB\n", device_get_free_memory(d));
}


