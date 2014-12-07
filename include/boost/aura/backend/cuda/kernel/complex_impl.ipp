// Complex numbers for both CUDA and OpenCL
//
// This file is included twice from complex.hpp, once with T defined as float
// and once as double.
//
// The macro magic in this file then creates the source for two complex number
// types "cfloat_t" and "cdouble_t" and their corresponding functions.
// Code in this is limited to C99, since OpenCL does not support C++ on
// devices.
//
#define HELPERCT1(x) c##x##_t
#define HELPERCT2(x) HELPERCT1(x)

#define HELPERCTROOT1(x) c##x
#define HELPERCTROOT2(x) HELPERCTROOT1(x)

#define CT HELPERCT2(T)
#define CTROOT HELPERCTROOT2(T)


#define M_CONC(A, B) M_CONC_(A, B)
#define M_CONC_(A, B) A##B


typedef struct {
    T real;
    T imag;
} CT;



  
AURA_DEVICE_FUNCTION T M_CONC(CTROOT,_real)(CT a) { return a.real; } 
AURA_DEVICE_FUNCTION T M_CONC(CTROOT,_imag)(CT a) { return a.imag; } 

AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_fromreal) (T a) { return (CT) { .real = a, .imag = 0};} 
AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_new)     (T a, T b) { return (CT) { .real = a, .imag =  b}; } 
AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_conj)    (CT a) { return (CT) { .real =a.real, .imag =  -a.imag}; } 

AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_add)(CT a, CT b) 
{ 
  return (CT) {.real = a.real + b.real, .imag = a.imag + b.imag}; 
} 


AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_addr)(CT a, T b) 
{ 
  return (CT) {.real = a.real + b, .imag = a.imag}; 
} 

AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_radd)(T a, CT b) 
{ 
  return (CT) {.real = a + b.real, .imag = b.imag}; 
} 

AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_mul)(CT a, CT b) 
{ 
  return (CT) {.real = a.real * b.real - a.imag * b.imag,
               .imag = a.real * b.imag + a.imag * b.real};
} 

AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_mulr)(CT a, T b) 
{ 
  return (CT) { .real = a.real * b, .imag = a.imag};
} 

AURA_DEVICE_FUNCTION CT M_CONC(CTROOT,_rmul)(T a, CT b) 
{ 
  return (CT) { .real = b.real * a, .imag = b.imag};
} 

#undef HELPERCT1
#undef HELPERCT2
#undef HELPERCTROOT1
#undef HELPERCTROOT2
#undef CT 
#undef CTROOT 
#undef M_CONC
#undef M_CONC_


