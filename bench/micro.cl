#include <boost/aura/backend.hpp>

__kernel void noarg() {}

__kernel void simple_add(__global float * A)
{
	int id = get_mesh_id();
	A[id] += 1.0;
}

__kernel void four_mad(__global float * A)
{
	int id = get_mesh_id();
	float f = A[id];
	f *= 41.0;
	f += 37.0;
	f *= 11.0;
	f += 23.0;
	f *= 2.0;
	f += 13.0;
	f *= 3.0;
	f += 7.0;
	A[id] = f;
}

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

__kernel void peak_flop(__global float * A)
{
	int id = get_mesh_id();
	float r0, r1, r2, r3, r4, r5, r6, r7;
	float r8, r9, r10, r11, r12, r13, r14, r15;
	r0 = 0.0001 * id;
	r1 = 0.0001 * id;
	r2 = 0.0002 * id;
	r3 = 0.0003 * id;
	r4 = 0.0004 * id;
	r5 = 0.0005 * id;
	r6 = 0.0006 * id;
	r7 = 0.0007 * id;
	r8 = 0.0008 * id;
	r9 = 0.0009 * id;
	r10 = 0.0010 * id;
	r11 = 0.0011 * id;
	r12 = 0.0012 * id;
	r13 = 0.0013 * id;
	r14 = 0.0014 * id;
	r15 = 0.0015 * id;

	for(int i=0; i<50; i++) {
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

__kernel void peak_flop_empty(__global float * A)
{
	int id = get_mesh_id();
	float r0, r1, r2, r3, r4, r5, r6, r7;
	float r8, r9, r10, r11, r12, r13, r14, r15;
	r0 = 0.0001 * id;
	r1 = 0.0001 * id;
	r2 = 0.0002 * id;
	r3 = 0.0003 * id;
	r4 = 0.0004 * id;
	r5 = 0.0005 * id;
	r6 = 0.0006 * id;
	r7 = 0.0007 * id;
	r8 = 0.0008 * id;
	r9 = 0.0009 * id;
	r10 = 0.0010 * id;
	r11 = 0.0011 * id;
	r12 = 0.0012 * id;
	r13 = 0.0013 * id;
	r14 = 0.0014 * id;
	r15 = 0.0015 * id;

	r0 += r1 + r2 + r3 + r4 + r5 + r6 + r7 +
	      r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
	A[id] = r0;
}

