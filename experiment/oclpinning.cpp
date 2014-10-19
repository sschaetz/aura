
#include <aura/backend.hpp>
#include <aura/config.hpp>
#include <aura/device_array.hpp>
#include <aura/misc/benchmark.hpp>

using namespace aura;
using namespace aura::backend;

const char * kernel_source =
"#include <aura/backend.hpp>\n"

"AURA_KERNEL void noarg() {} "

"AURA_KERNEL void simple_add(AURA_GLOBAL float * A) {"
"	int id = get_mesh_id();"
"	A[id] += 1.0;"
"}";

template <typename T>
T * opencl_malloc_pinned(std::size_t num, device & d, cl_mem* mem)
{
	std::size_t bytes = num*sizeof(T);
	cl_int errorcode;
	*mem = clCreateBuffer(d.get_backend_context(),
			CL_MEM_ALLOC_HOST_PTR, bytes, NULL,
			&errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	feed f(d);
	void * v = clEnqueueMapBuffer(f.get_backend_stream(),
			*mem, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE,
			0, bytes, 0, NULL, NULL, &errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	printf("m: %p\n", *mem);
	printf("v: %p\n", v);

	AURA_OPENCL_SAFE_CALL(clEnqueueUnmapMemObject(
				f.get_backend_stream(), 
				*mem, (void*)v, 0, NULL, NULL));
	return (T*)v;
}

void opencl_copy_pinned(cl_mem dst, cl_mem src, 
		std::size_t num, feed& f)
{
	AURA_OPENCL_SAFE_CALL(
			clEnqueueCopyBuffer(f.get_backend_stream(),
				src, dst, 0, 0, num, 0, NULL, NULL));
}
 

// if this is used, is a memory copy from the pointer fast or from
// the memory object m?
// is anything allocated (on the device or on the host?)
// how can this resource be freed up again?
// do we have to have the m object to free this thing?
// if the buffer is freed, is p freed?
void * opencl_pin(void * p, std::size_t bytes, device & d)
{
	cl_int errorcode;
	cl_mem m = clCreateBuffer(d.get_backend_context(),
			CL_MEM_USE_HOST_PTR, bytes, p,
			&errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	feed f(d);
	void * v = clEnqueueMapBuffer(f.get_backend_stream(),
			m, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE,
			0, bytes, 0, NULL, NULL, &errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	return (void*)v;
}

int main(void)
{
	initialize();
	int num = device_get_count();
	
	AURA_CHECK_ERROR(0 > num)
	device d(0);
	feed f(d);

	// theory: if we transfer large amounts of data from this
	// pointer, it should be faster

	// 32M floats 
	std::size_t size = 1024*1024*32;
	cl_mem hostmembuffer;
	(void)opencl_malloc_pinned<float>(size, 
			d, &hostmembuffer);
	float* hostmem = (float*)malloc(sizeof(float)*size);
	device_ptr<float> devmem = device_malloc<float>(size, d);

	// benchmark result variables
	double min, max, mean, stdev;
	std::size_t runs;	

	AURA_BENCHMARK_ASYNC(copy(devmem, hostmem, size, f), wait_for(f), 
			2*1e6, min, max, mean, stdev, runs);
	print_benchmark_results("normal", min, max, mean, stdev, runs, 2*1e6);
	AURA_BENCHMARK_ASYNC(opencl_copy_pinned(devmem.get(), hostmembuffer, 
		sizeof(float)*size, f), 
			wait_for(f), 2*1e6, min, max, mean, stdev, runs);
	print_benchmark_results("pinned", min, max, mean, stdev, runs, 2*1e6);
	
	wait_for(f);

}


