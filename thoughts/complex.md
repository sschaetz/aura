# Complex

We need c-style complex functions. What is the best naming
scheme for these functions?

## Christian proposes (I think):
cfloat_real()
cfloat_new()
cdouble_add()
etc.

## Netlib has:
c_abs()
z_abs()

## CUDA has:
cuCabs
cuCabsf

## MKL has:
v[m]<?><name><mod>
so: 
vsAbs
vdAbs
vcAbs
vzAbs
with: s REAL d DOUBLE PRECISION c COMPLEX z DOUBLE COMPLEX 

## C standard:
crealf
creal
creall

## GSL has:
gsl_complex_add (double only in GSL)

##  John Burkardt has:
C4_ABS
C8_ABS

## Other Alternatives:
cf_real()
cd_real()
real_cd()
real_cf()
crealf
creald
realcf
realcd

cfloat
cdouble
c4
c8
cs
cd
cf

so:
cfloat, cdouble
and:
crealf, creal

