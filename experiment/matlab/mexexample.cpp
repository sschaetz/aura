#include <complex>

#include <boost/shared_ptr.hpp>

#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/misc/matlab.hpp>

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
};

// if this goes out of scope, state is destroyed, must be global
boost::shared_ptr<state_t> state_ptr;

void init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	initialize();
	state_ptr = boost::shared_ptr<state_t>(new state_t);

	// convenience reference to avoid pointer semantics
	state_t& state = *state_ptr;
	state.d = device(0);
	state.f = feed(state.d);
    
    state.m = create_module_from_file("kernel.ptx", state.d);
	state.c2i = create_kernel(state.m, "kern_c2i");
	state.i2c = create_kernel(state.m, "kern_i2c");
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
	
	if (!strcmp("finish", cmd)) {
		finish(nlhs, plhs, nrhs, prhs);
		return;
	}

	mexErrMsgTxt("invalid command string");
}

