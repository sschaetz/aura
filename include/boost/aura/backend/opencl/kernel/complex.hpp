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
	float n = 1.0f / (y.x*y.x + y.y*y.y);
	float r = (x.x*y.x + x.y*y.y) * n;
	float i = (x.y*y.x - x.x*y.y) * n;
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
	if (s == 0.) {
		return 0.;
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
	double n = 1.0f / (y.x*y.x + y.y*y.y);
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

