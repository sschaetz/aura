#ifndef AURA_BACKEND_OPENCL_KERNEL_COMPLEX_HPP
#define AURA_BACKEND_OPENCL_KERNEL_COMPLEX_HPP

// single precision -----
typedef float2 cfloat;

inline float crealf(cfloat x) 
{ 
	return x.x; 
}

inline float cimagf(cfloat x) 
{ 
	return x.y; 
}


__global float *crealfp( __global cfloat * x)
{
	return ( __global float *) x;
}

__global float *cimagfp(__global cfloat *x)
{
	return ( (__global float *) x + 1);
}


inline cfloat make_cfloat(float r, float i)
{
	return (cfloat)(r, i);
}

inline cfloat conjf(cfloat x)
{
    return make_cfloat(x.x, -x.y);
}

inline cfloat caddf(cfloat x, cfloat y)
{
	return make_cfloat(x.x+y.x, x.y+y.y);
}

inline cfloat csubf(cfloat x, cfloat y)
{
	return make_cfloat(x.x-y.x, x.y-y.y);
}

inline cfloat cmulf(cfloat x, cfloat y)
{
	return make_cfloat(x.x * y.x - x.y * y.y, 
			x.x*y.y + x.y*y.x);
}

inline cfloat cdivf(cfloat x, cfloat y)
{	
	// This implementation doesn't function well for large numbers
	// float n = 1.0f / (y.x*y.x + y.y*y.y);
	// float r = (x.x*y.x + x.y*y.y) * n;
	// float i = (x.y*y.x - x.x*y.y) * n;
	
	// The following implementation is taken from the LLVM Compiler Infrastructure, licensed under 
	// the MIT and the University of Illinois Open Source Licenses.
	// https://code.openhub.net/file?fid=XBgmMXzw1oxpd_pKEX4Olpef3gM&cid=DwH1iTUyTao&s=__divsc3&fp=406477&mp&projSelected=true#L0
	
	float a = x.x;
	float b = x.y;
	float c = y.x;
	float d = y.y;
	
	int ilogbw = 0;
    float logbw = logb(fmax(fabs(c), fabs(d)));    
    ilogbw = (int)logbw;
    c = ldexp(c, -ilogbw);
    d = ldexp(d, -ilogbw);
    
    float denom = 1.0f / (c * c + d * d);
    float r = ldexp((a * c + b * d) * denom, -ilogbw);
    float i = ldexp((b * c - a * d) * denom, -ilogbw);		
	
	return make_cfloat(r, i);
}

inline float cabsf(cfloat x)
{
	float rp = x.x;
	float ip = x.y;
	float s = fabs(rp);
	if (s < fabs(ip)) {
		s = fabs(ip);
	}	
	if (s == 0.f) {
		return 0.f;
	}
	rp /= s;
	ip /= s;

	return s * sqrt(rp * rp + ip * ip);
}

#ifdef AURA_KERNEL_ENABLE_DOUBLE
// double precision -----
typedef double2 cdouble;

inline double creal(cdouble x) 
{ 
	return x.x; 
}

inline double cimag(cdouble x) 
{ 
	return x.y; 
}

__global double *crealp( __global cdouble * x)
{
	return ( __global double *) x;
}

__global double*cimagp(__global cdouble *x)
{
	return ( (__global double *) x + 1);
}

inline cdouble make_cdouble(double r, double i)
{
	return (cdouble)(r, i);
}

inline cdouble conj(cdouble x)
{
	return make_cdouble(x.x, -x.y);
}

inline cdouble cadd(cdouble x, cdouble y)
{
	return make_cdouble(x.x+y.x, x.y+y.y);
}

inline cdouble csub(cdouble x, cdouble y)
{
	return make_cdouble(x.x-y.x, x.y-y.y);
}

inline cdouble cmul(cdouble x, cdouble y)
{
	return make_cdouble(x.x * y.x - x.y * y.y, 
			x.x*y.y + x.y*y.x);
}

inline cdouble cdiv(cdouble x, cdouble y)
{
	double n = 1.0 / (y.x*y.x + y.y*y.y);
	double r = (x.x*y.x + x.y*y.y) * n;
	double i = (x.y*y.x - x.x*y.y) * n;
	return make_cdouble(r, i);
}

inline double cabs(cdouble x)
{	
	double rp = x.x;
	double ip = x.y;
	double s = fabs(rp);
	if (s < fabs(ip)) {
		s = fabs(ip);
	}	
	if (s == 0.) {
		return 0.;
	}
	rp /= s;
	ip /= s;

	return s * sqrt(rp * rp + ip * ip);
}

/// promotion
inline cdouble cfloat_to_cdouble(cfloat c)
{
    return make_cdouble((double)crealf(c), (double)cimagf(c));
}

/// demotion
inline cfloat cdouble_to_cfloat(cdouble c) 
{
	return make_cfloat((float)creal(c), (float)cimag(c));
}

#endif // AURA_KERNEL_ENABLE_DOUBLE

#endif // AURA_BACKEND_OPENCL_KERNEL_COMPLEX_HPP

