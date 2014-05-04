#include <aura/backend.hpp>
#define PEAK_FLOP_INIT(T) \
  T r0 = 0.0001 * id;     \
  T r1 = 0.0001 * id;     \
  T r2 = 0.0002 * id;     \
  T r3 = 0.0003 * id;     \
  T r4 = 0.0004 * id;     \
  T r5 = 0.0005 * id;     \
  T r6 = 0.0006 * id;     \
  T r7 = 0.0007 * id;     \
  T r8 = 0.0008 * id;     \
  T r9 = 0.0009 * id;     \
  T r10 = 0.0010 * id;    \
  T r11 = 0.0011 * id;    \
  T r12 = 0.0012 * id;    \
  T r13 = 0.0013 * id;    \
  T r14 = 0.0014 * id;    \
  T r15 = 0.0015 * id;    \
  /**/

#define PEAK_FLOP_MADD \
  r0 = r1*r8+r0;       \
  r1 = r15*r9+r2;      \
  r2 = r14*r10+r4;     \
  r3 = r13*r11+r6;     \
  r4 = r12*r12+r8;     \
  r5 = r11*r13+r10;    \
  r6 = r10*r14+r12;    \
  r7 = r9*r15+r14;     \
  r8 = r7*r0+r1;       \
  r9 = r8*r1+r3;       \
  r10 = r6*r2+r5;      \
  r11 = r5*r3+r7;      \
  r12 = r4*r4+r9;      \
  r13 = r3*r5+r11;     \
  r14 = r2*r6+r13;     \
  r15 = r0*r7+r15;     \
  /**/

AURA_KERNEL void peak_flop_single(AURA_GLOBAL float * A) {
  int id = get_mesh_id();
  PEAK_FLOP_INIT(float);
  for(int i=0; i<64; i++) {
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
  }
  r0 += r1 + r2 + r3 + r4 + r5 + r6 + r7 + 
    r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
  A[id] = r0;
}

AURA_KERNEL void peak_flop_double(AURA_GLOBAL float * A) {
  int id = get_mesh_id();
  PEAK_FLOP_INIT(double);
  for(int i=0; i<64; i++) {
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
    PEAK_FLOP_MADD;
  }
  r0 += r1 + r2 + r3 + r4 + r5 + r6 + r7 + 
    r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
  A[id] = r0;
}

#define BSIZE 16

AURA_KERNEL void peak_copy(AURA_GLOBAL float * dst, AURA_GLOBAL float* src) {
  const int bsize = BSIZE;
  const int mult = 128;
  int id = (get_mesh_id() / bsize)*bsize*mult + get_mesh_id() % bsize; 
#pragma unroll
  for(int i=0; i<mult; i++) {
    ds[id + i * bsize] = src[id + i * bsize];
  }
}

AURA_KERNEL void peak_scale(AURA_GLOBAL float * dst, AURA_GLOBAL float * src,
  float scalar) {
  const int bsize = BSIZE;
  const int mult = 64;
  int id = (get_mesh_id() / bsize)*bsize*mult + get_mesh_id() % bsize; 
  for(int i=0; i<mult; i++) {
    dst[id + i * bsize] = scalar * src[id + i * bsize];
  }
}

AURA_KERNEL void peak_add(AURA_GLOBAL float * dst, AURA_GLOBAL float * src1,
  AURA_GLOBAL float * src2) {
  const int bsize = BSIZE;
  const int mult = 64;
  int id = (get_mesh_id() / bsize)*bsize*mult + get_mesh_id() % bsize; 
  for(int i=0; i<mult; i++) {
    dst[id + i * bsize] = src1[id + i * bsize] * src2[id + i * bsize];
  }
}

AURA_KERNEL void peak_triad(AURA_GLOBAL float * dst, AURA_GLOBAL float * src1,
  AURA_GLOBAL float * src2, float scalar) {
  const int bsize = BSIZE;
  const int mult = 64;
  int id = (get_mesh_id() / bsize)*bsize*mult + get_mesh_id() % bsize; 
  for(int i=0; i<mult; i++) {
    dst[id + i * bsize] = src1[id + i * bsize] + scalar * src2[id + i * bsize];
  }
}

