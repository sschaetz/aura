#include <complex>

#include <boost/shared_ptr.hpp>

#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/misc/matlab.hpp>

#include "mex.h"
#include "matrix.h"

using namespace boost::aura::backend;
using namespace boost::aura;


// this type holds the state of the mex file
struct state_t
{	
    ~state_t()
    {
        printf("OH NO IM DEAD!\n");
    }
    device d;
	feed f;

	// variable vectors
	device_array<float> v;
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
}


void compute(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    state_t& state = *state_ptr;
    std::cout << get_bounds(prhs[0]) << std::endl;
    
    // device memory
    state.v = device_array<float>(get_bounds(prhs[0]), state.d);
    
    // matlab output
    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]), 
            mxGetDimensions(prhs[0]), mxSINGLE_CLASS, mxREAL);
    
    // upload to device
    copy(reinterpret_cast<float*>(mxGetData(prhs[0])), state.v, state.f);
    
    // download from device
    copy(state.v, reinterpret_cast<float*>(mxGetData(plhs[0])), state.f);
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

