#include <aura/backend.hpp>

#define PEAK_FLOP_INIT(T) \
  T r0 = 0.0000001 * id;     \
  T r1 = 0.0000001 * id;     \
  T r2 = 0.0000002 * id;     \
  T r3 = 0.0000003 * id;     \
  T r4 = 0.0000004 * id;     \
  T r5 = 0.0000005 * id;     \
  T r6 = 0.0000006 * id;     \
  T r7 = 0.0000007 * id;     \
  T r8 = 0.0000008 * id;     \
  T r9 = 0.0000009 * id;     \
  T r10 = 0.0000010 * id;    \
  T r11 = 0.0000011 * id;    \
  T r12 = 0.0000012 * id;    \
  T r13 = 0.0000013 * id;    \
  T r14 = 0.0000014 * id;    \
  T r15 = 0.0000015 * id;    \
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

AURA_KERNEL void peak_copy(AURA_GLOBAL float * dst, AURA_GLOBAL float * src) {
  int id = get_mesh_id();
  int s = get_mesh_size();
#pragma unroll
  for(int i=0; i<64; i++) {
    dst[id] = src[id];
    id += s;
  }
}

AURA_KERNEL void peak_scale(AURA_GLOBAL float * dst, AURA_GLOBAL float * src,
  float scalar) {
  int id = get_mesh_id();
  int s = get_mesh_size();
  for(int i=0; i<64; i++) {
    dst[id] = scalar * src[id];
    id += s;
  }
}

AURA_KERNEL void peak_add(AURA_GLOBAL float * dst, AURA_GLOBAL float * src1,
  AURA_GLOBAL float * src2) {
  int id = get_mesh_id();
  int s = get_mesh_size();
  for(int i=0; i<64; i++) {
    dst[id] = src1[id] + src2[id];
    id += s;
  }
}

AURA_KERNEL void peak_triad(AURA_GLOBAL float * dst, AURA_GLOBAL float * src1,
  AURA_GLOBAL float * src2, float scalar) {
  int id = get_mesh_id();
  int s = get_mesh_size();
  for(int i=0; i<64; i++) {
    dst[id] = src1[id] + scalar * src2[id];
    id += s;
  }
}

