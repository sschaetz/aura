#include <complex>

#include <boost/shared_ptr.hpp>
#include "nullptr.hpp"

#include <aura/backend.hpp>
#include <aura/device_array.hpp>
#include <aura/copy.hpp>
#include "mex.h"
#include "matrix.h"

using namespace aura::backend;
using namespace aura;

#define USE_SINGLE

#ifdef USE_SINGLE

typedef float float_t;
void check_type(const mxArray* pm) 
{
	if(!mxIsSingle(pm)) {
		mexErrMsgTxt("double is not supported");
	}
}

#else

typedef double float_t;

void check_type(const mxArray* pm) 
{
	if(!mxIsDouble(pm)) {
		mexErrMsgTxt("singleis not supported");
	}
}

#endif


typedef std::complex<float_t> cfloat_t;




// matlab helper functions -----

/// get bounds of mxArray
bounds get_bounds(const mxArray* pm) 
{
	const mwSize numdims = mxGetNumberOfDimensions(pm);
	const mwSize * dims = mxGetDimensions(pm);
	bounds b;
	for (int i=0; i<numdims; i++) {
		b.push_back((std::size_t)dims[i]);
	}
	return b;
}

/// check rank must be in [min, max[
void check_rank(const mxArray* pm, mwSize min, mwSize max = -1) 
{
	if (-1 == max) {
		max = min+1;
	}
	mwSize rank = mxGetNumberOfDimensions(pm);
	if (min <= rank && max > rank) {
		mexErrMsgTxt("incorrect rank");
	}
}
// -----




// this type holds the state of the mex file
struct state_t
{
	device d;
	feed f;

	// kernels
	module kernel_module;
	kernel kernel_c2i;
	kernel kernel_i2c;

	// constant vectors 
	device_array<cfloat_t> c;
	device_array<cfloat_t> m;
	device_array<cfloat_t> p;

	// variable vectors
	device_array<float_t> dm;
	device_array<float_t> res;
};

// if this goes out of scope, state is destroyed, must be global
boost::shared_ptr<state_t> state_ptr;

void init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	state_ptr = boost::shared_ptr<state_t>(new state_t);

	// convenience reference to avoid pointer semantics
	state_t& state = *state_ptr;
	state.d = device(0);
	state.f = feed(state.d);

	// create kernel functions
	state.kernel_module = create_module_from_file("mexexample_kernel.cu", state.d);
	state.kernel_c2i = create_kernel(state.kernel_module, "kernel_c2i");
	state.kernel_i2c = create_kernel(state.kernel_module, "kernel_i2c");

	// check the parameters, we need 3 3-dimensional complex vectors
	if (3 != nrhs) {
	    mexErrMsgTxt("init requires 3 arguments");
	}
	
	// check for complex values (we don't support real FFTs)
	// check for correct type 
	for (int i=0; i<3; i++) {
		if(!mxIsComplex(prhs[i])) {
			mexErrMsgTxt("real is not supported");
		}
		check_type(prhs[i]);
	}
	
	// initialize c
	check_rank(prhs[0], 2, 4);	
	state.c = device_array<cfloat_t>(get_bounds(prhs[0]), state.d);
	copy(state.c, reinterpret_cast<cfloat_t*>(mxGetData(prhs[0])), state.f);

	// initialize mask
	check_rank(prhs[1], 2);	
	state.m = device_array<cfloat_t>(get_bounds(prhs[1]), state.d);
	copy(state.m, reinterpret_cast<cfloat_t*>(mxGetData(prhs[1])), state.f);

	// initialize p
	check_rank(prhs[2], 3);	
	state.p = device_array<cfloat_t>(get_bounds(prhs[2]), state.d);
	copy(state.p, reinterpret_cast<cfloat_t*>(mxGetData(prhs[2])), state.f);
}


void compute(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
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

