#ifndef AURA_BACKEND_OPENCL_KERNEL_COMPLEX_HPP
#define AURA_BACKEND_OPENCL_KERNEL_COMPLEX_HPP

// single precission -----
typedef float2 cfloat;

__device__ static __forceinline__ float crealf(cfloat x) 
{ 
	return x.x; 
}

__device__ static __forceinline__ float cimagf(cfloat x) 
{ 
	return x.y; 
}

__device__ static __forceinline__ cfloat make_cfloat(float r, float i)
{
	return make_cuFloatComplex(r, i);
}

__device__ static __forceinline__ cfloat conjf(cfloat x)
{
    return make_cfloat(x.x, -x.y);
}

__device__ static __forceinline__ cfloat caddf(cfloat x, cfloat y)
{
	return make_cfloat(x.x+y.x, x.y+y.y);
}

__device__ static __forceinline__ cfloat csubf(cfloat x, cfloat y)
{
	return make_cfloat(x.x-y.x, x.y-y.y);
}

__device__ static __forceinline__ cfloat cmulf(cfloat x, cfloat y)
{
	return make_cfloat(x.x * y.x - x.y * y.y, 
			x.x*y.y + x.y*y.x);
}

__device__ static __forceinline__ cfloat cdivf(cfloat x, cfloat y)
{
	float n = x.y*x.y + y.y * y.y;
	float r = (x.x*y.x + x.y*y.y) / n;
	float i = (x.y-y.x + x.x*y.y) / n;
	return make_cfloat(r, i);
}

__device__ static __forceinline__ float cabsf(cfloat x)
{
	float s = x.x > x.y ? x.x : x.y;
	if (s == 0.) {
		return 0.;
	}
	return s * sqrt(x.x * x.x + x.y * x.y);
}

__device__ static __forceinline__ cuComplex cfmaf(cfloat x, cfloat y, cfloat d)
{
	return cuCfmaf(x, y, d);
}

// double precission -----
typedef cuDoubleComplex cdouble;

__device__ static __forceinline__ double creal(cdouble x) 
{ 
	return x.x; 
}

__device__ static __forceinline__ double cimag(cdouble x) 
{ 
	return x.y; 
}

__device__ static __forceinline__ cdouble make_cdouble(double r, double i)
{
	return make_cuDoubleComplex(r, i);
}

__device__ static __forceinline__ cdouble conj(cdouble x)
{
	return cuConj(x);
}

__device__ static __forceinline__ cdouble cadd(cdouble x, cdouble y)
{
	return cuCadd(x, y);
}

__device__ static __forceinline__ cdouble csub(cdouble x, cdouble y)
{
	return cuCsub(x, y);
}

__device__ static __forceinline__ cdouble cmul(cdouble x, cdouble y)
{
	return cuCmul(x, y);
}

__device__ static __forceinline__ cdouble cdiv(cdouble x, cdouble y)
{
	return cuCdiv(x, y);
}

__device__ static __forceinline__ double cabs(cdouble x)
{
	return cuCabs(x);
}

__device__ static __forceinline__ cdouble cfma(cdouble x, cdouble y, cdouble d)
{
	return cuCfma(x, y, d);
}


/// promotion
__device__ static __forceinline__ cdouble cfloat_to_cdouble(cfloat c)
{
    return make_cdouble((double)crealf(c), (double)cimagf(c));
}

/// demotion
__device__ static __forceinline__ cfloat cdouble_to_cfloat(cdouble c) 
{
	return make_cfloat((float)creal(c), (float)cimag(c));
}

#endif // AURA_BACKEND_OPENCL_KERNEL_COMPLEX_HPP

