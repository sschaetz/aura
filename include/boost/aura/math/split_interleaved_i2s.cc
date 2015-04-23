#include <boost/aura/backend.hpp>

AURA_KERNEL void s2i_float(AURA_GLOBAL float* real_part,
    AURA_GLOBAL float* imag_part,
    AURA_GLOBAL cfloat* dst, unsigned long count)
{
  unsigned int i = get_mesh_id();
  if (i < count) {
    dst[i] = make_cfloat(real_part[i], imag_part[i]);
  }
}

AURA_KERNEL void i2s_float(AURA_GLOBAL cfloat* src,
    AURA_GLOBAL float* real_part,
    AURA_GLOBAL float* imag_part, unsigned long count)
{
  unsigned int i = get_mesh_id();
  if (i < count) {
    real_part[i] = crealf(src[i]);
    imag_part[i] = cimagf(src[i]);
  }
}

