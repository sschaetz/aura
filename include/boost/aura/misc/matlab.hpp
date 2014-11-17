#ifndef AURA_MISC_MATLAB_HPP
#define AURA_MISC_MATLAB_HPP

#include "matrix.h"

namespace boost
{
namespace aura 
{
namespace matlab 
{

/// check if mxArray contains 32bit floating point data 
bool is_single(const mxArray* pm) 
{
	return mxIsSingle(pm);
}

/// check if mxArray contains 64bit floating point data 
bool is_double(const mxArray* pm) 
{
	return mxIsDouble(pm);
}

/// check if mxArray contains complex data 
bool is_complex(const mxArray* pm) 
{
	return mxIsComplex(pm);
}

/// check if mxArray contains real data 
bool is_real(const mxArray* pm) 
{
	return !is_complex(pm);	
}

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

/// check if rank of mxArray is in [min, max[
bool check_rank(const mxArray* pm, mwSize min, mwSize max = -1) 
{
	if (-1 == max) {
		max = min+1;
	}
	mwSize rank = mxGetNumberOfDimensions(pm);
	if (min <= rank && max > rank) {
		return false;
	}
	return true;
}



} // namespace matlab 
} // namespace aura
} // namespace boost 

#endif // AURA_MISC_PROFILE_HPP

