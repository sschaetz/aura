#include <complex>

#include <boost/shared_ptr.hpp>

#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/misc/matlab.hpp>
#include <boost/aura/misc/benchmark.hpp>

#include "mex.h"
#include "matrix.h"

using namespace boost::aura::backend;
using namespace boost::aura::matlab;
using namespace boost::aura;

typedef std::complex<float> cfloat;

// this type holds the state of the mex file
struct state_t
{	
    ~state_t()
    {
        printf("OH NO IM DEAD!\n");
    }
    device d;
	feed f;

	// vectors
	device_array<float> ior;
    device_array<float> ioi;
    device_array<cfloat> ioc;

    // module and kernels
    module m;
    kernel c2i;
    kernel i2c;
    kernel axpy;
};

// if this goes out of scope, state is destroyed, must be global
boost::shared_ptr<state_t> state_ptr;

void init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	initialize();
	state_ptr = boost::shared_ptr<state_t>(new state_t);

	// convenience reference to avoid pointer semantics
	state_t& state = *state_ptr;
	state.d = device(1);
	state.f = feed(state.d);
    
    state.m = create_module_from_file("kernel.ptx", state.d);
	state.c2i = create_kernel(state.m, "kern_c2i");
	state.i2c = create_kernel(state.m, "kern_i2c");
    state.axpy = create_kernel(state.m, "kern_axpy");
}


void compute(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    state_t& state = *state_ptr;
    std::cout << get_bounds(prhs[0]) << std::endl;
    
    if(!is_single(prhs[0]) ||  !is_complex(prhs[0])) {
        mexErrMsgTxt("invalid type, expecting complex single");
    }
    
    // device memory
    state.ior = device_array<float>(get_bounds(prhs[0]), state.d);
    state.ioi = device_array<float>(get_bounds(prhs[0]), state.d);
    state.ioc = device_array<cfloat>(get_bounds(prhs[0]), state.d);

    // matlab output
    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]), 
            mxGetDimensions(prhs[0]), mxSINGLE_CLASS, mxCOMPLEX);
    
    // upload to device
    copy(reinterpret_cast<float*>(mxGetData(prhs[0])), state.ior, state.f);
    copy(reinterpret_cast<float*>(mxGetImagData(prhs[0])), 
            state.ioi, state.f);

    // c2i
    invoke(state.c2i, state.ioc.get_bounds(), 
		args(state.ioc.size(), 
            state.ior.begin_ptr(), state.ioi.begin_ptr(), 
            state.ioc.begin_ptr()), 
            state.f);
        
    // i2c
    invoke(state.i2c, state.ioc.get_bounds(), 
		args(state.ioc.size(), 
            state.ioc.begin_ptr(),
            state.ior.begin_ptr(), state.ioi.begin_ptr()), 
            state.f);
    
    // download from device
    copy(state.ior, reinterpret_cast<float*>(mxGetData(plhs[0])), state.f);
    copy(state.ioi, reinterpret_cast<float*>(mxGetImagData(plhs[0])), 
            state.f);

    wait_for(state.f);

}

void axpy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    state_t& state = *state_ptr;
    
    // device memory
    state.ior = device_array<float>(get_bounds(prhs[1]), state.d);
    state.ioi = device_array<float>(get_bounds(prhs[1]), state.d);
    device_array<cfloat> X(get_bounds(prhs[1]), state.d);
    device_array<cfloat> Y(get_bounds(prhs[2]), state.d);

    std::cout << get_bounds(prhs[1]) << " " << get_bounds(prhs[2]) << std::endl;
    std::cout << "device memory allocated" << std::endl;
    
    // matlab output
    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[2]), 
            mxGetDimensions(prhs[2]), mxSINGLE_CLASS, mxCOMPLEX);
    
    cfloat alpha(*reinterpret_cast<float*>(mxGetData(prhs[0])), 
            *reinterpret_cast<float*>(mxGetImagData(prhs[0])));
    std::cout << "alpha: " << alpha << std::endl;
    
    // upload X to device
    copy(reinterpret_cast<float*>(mxGetData(prhs[1])), state.ior, state.f);
    copy(reinterpret_cast<float*>(mxGetImagData(prhs[1])), 
            state.ioi, state.f);
    // c2i
    invoke(state.c2i, X.get_bounds(), 
		args(X.size(), 
            state.ior.begin_ptr(), state.ioi.begin_ptr(), 
            X.begin_ptr()), 
            state.f);
    
    // upload Y to device
    copy(reinterpret_cast<float*>(mxGetData(prhs[2])), state.ior, state.f);
    copy(reinterpret_cast<float*>(mxGetImagData(prhs[2])), 
            state.ioi, state.f);
    // c2i
    invoke(state.c2i, Y.get_bounds(), 
		args(Y.size(), 
            state.ior.begin_ptr(), state.ioi.begin_ptr(), 
            Y.begin_ptr()), 
            state.f);

    
    std::cout << "befor done" << std::endl;

    
    // AXPY
    invoke(state.axpy, X.get_bounds(), 
		args(X.size(), alpha, X.begin_ptr(), Y.begin_ptr()), 
            state.f);
    
    benchmark_result br;
   
    std::cout << "axpy done" << std::endl;
    
    // i2c
    invoke(state.i2c, Y.get_bounds(), 
		args(Y.size(), 
            Y.begin_ptr(),
            state.ior.begin_ptr(), state.ioi.begin_ptr()), 
            state.f);
    
    // download from device
    copy(state.ior, reinterpret_cast<float*>(mxGetData(plhs[0])), state.f);
    copy(state.ioi, reinterpret_cast<float*>(mxGetImagData(plhs[0])), 
            state.f);

    wait_for(state.f);
    
    AURA_BENCH_ASYNC(
        invoke(state.axpy, X.get_bounds(), 
            args(X.size(), alpha, X.begin_ptr(), Y.begin_ptr()), 
                state.f);, wait_for(state.f), 1000000, br);
    std::cout << "benchmark result" << std::endl;
    std::cout << br << std::endl;

}

void finish(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	state_ptr.reset();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// get the command string
	char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd))) {
		mexErrMsgTxt("missing or invalid command string");
	}

	// discard first right hand side argument
	nrhs--;
	prhs++;

	if (!strcmp("init", cmd)) {
		init(nlhs, plhs, nrhs, prhs);
		return;
	}
	
	// check if state is initialized
	if (0 == state_ptr.use_count()) {
		mexErrMsgTxt("not initialized, call init first");
	}

	if (!strcmp("compute", cmd)) {
		compute(nlhs, plhs, nrhs, prhs);
		return;
	}
    
    if (!strcmp("axpy", cmd)) {
		axpy(nlhs, plhs, nrhs, prhs);
		return;
	}
	
	if (!strcmp("finish", cmd)) {
		finish(nlhs, plhs, nrhs, prhs);
		return;
	}

	mexErrMsgTxt("invalid command string");
}

