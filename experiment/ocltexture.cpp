#include <iostream>
#include <algorithm>
#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/copy.hpp>

#ifdef __APPLE__
	#include "OpenCL/cl.hpp"
#else
	#include "CL/cl.h"
#endif


using namespace boost;

const char* manual_interp_kernel_str = R"kernel_str(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void manual_interp(AURA_GLOBAL float* src,
			AURA_GLOBAL float* dst,
			float interp,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N-1) {
			dst[i] = src[i] * (1.-interp) + interp*src[i+1];
		}
	}
		
		)kernel_str";


const char* tex_interp_kernel_str = R"kernel_str(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void tex_interp(__read_only image1d_buffer_t src,
			AURA_GLOBAL float* dst,
			float interp,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		const sampler_t smp = CLK_FILTER_LINEAR | 
			CLK_NORMALIZED_COORDS_FALSE | 
			CLK_ADDRESS_CLAMP;
		if (i < N && i > 0) {
			interp += i;
			float4 r = read_imagef(src, (int)interp);
			dst[i] = r.x;
		}
	}
		
		)kernel_str";




	

int main(void)
{
	aura::initialize();
	aura::device d(2);
	aura::feed f(d);

	std::size_t s = 64;
	std::vector<float> in(s);
	std::vector<float> out(s, 0.);
	float start = 1.;
	std::generate(in.begin(), in.end(), [&]() {
				return start++;
			});
	aura::device_array<float> din(s, d);
	aura::device_array<float> dout(s, d);
	aura::copy(in, din, f);
	wait_for(f);
	aura::copy(out, dout, f);
	float interp = 0.600;
	auto manual_interp = d.load_from_string(
			"manual_interp",
			manual_interp_kernel_str,
			AURA_BACKEND_COMPILE_FLAGS, true);
	auto tex_interp = d.load_from_string(
			"tex_interp",
			tex_interp_kernel_str,
			AURA_BACKEND_COMPILE_FLAGS, true);


	// create an image
	
	cl_image_format format;
	format.image_channel_order = CL_R;
	format.image_channel_data_type = CL_FLOAT;
	
        cl_image_desc desc =
        {
		CL_MEM_OBJECT_IMAGE1D,
		in.size(),
		0, 0, 0, 0, 0, 0, 0,
		NULL
		//din.begin().get()
        };

	wait_for(f);
	int errorcode = 0;
	cl_mem dtexin = clCreateImage(
			d.get_backend_context(), 
			CL_MEM_READ_ONLY, 
			&format,
			&desc,
			NULL,
			&errorcode);
	wait_for(f);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	
	size_t dst_origin[3];
	dst_origin[0] = 0;
	dst_origin[1] = 0;
	dst_origin[2] = 0;
	size_t region[3];
	region[0] = in.size();
	region[1] = 1;
	region[2] = 1;

	AURA_OPENCL_SAFE_CALL(clEnqueueCopyBufferToImage(
				f.get_backend_stream(),
				din.begin().get(),
				dtexin,
				0,
				dst_origin,
				region,
				0,
				NULL,
				NULL));


	// test kernels
	#if 0
	aura::invoke(manual_interp, 
			aura::bounds(in.size()), 
			aura::args(din.begin_ptr(), 
				dout.begin_ptr(), 
				interp, 
				in.size()
			), f
		);
	#endif
	wait_for(f);

	aura::invoke(tex_interp, 
			aura::bounds(in.size()), 
			aura::args(dtexin, 
				dout.begin_ptr(), 
				interp, 
				in.size()
			), f
		);
	wait_for(f);

	aura::copy(dout, out, f);
	aura::wait_for(f);

	for (auto x : in) {
		std::cout << x << " ";
	}
	std::cout << std::endl;
	for (auto x : out) {
		std::cout << x << " ";
	}
	std::cout << std::endl;
}

