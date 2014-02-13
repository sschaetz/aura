
#include <aura/backend.hpp>
#include <aura/config.hpp>
#include <aura/device_array.hpp>

using namespace aura;
using namespace aura::backend;

const char * kernel_source =
"#include <aura/backend.hpp>\n"

"AURA_KERNEL void noarg() {} "

"AURA_KERNEL void simple_add(AURA_GLOBAL float * A) {"
"	int id = get_mesh_id();"
"	A[id] += 1.0;"
"}";

void * opencl_malloc_pinned(std::size_t bytes, device & d)
{
	cl_int errorcode;
	cl_mem m = clCreateBuffer(d.get_backend_context(),
			CL_MEM_ALLOC_HOST_PTR, bytes, NULL,
			&errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	feed f(d);
	void * v = clEnqueueMapBuffer(f.get_backend_stream(),
			m, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE,
			0, bytes, 0, NULL, NULL, &errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	printf("m: %lu\n", m);
	printf("v: %lu\n", v);
	return (void*)m;
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

void * opencl_free_pinned(void * ptr) {
	// here we need m?
	// that does not make any sense
}

int main(void)
{
	initialize();
	int num = device_get_count();

	AURA_CHECK_ERROR(0 < num)
	device d(0);
	feed f(d);
	// theory: if we transfer large amounts of data from this
	// pointer, it should be faster
	void * x = opencl_malloc_pinned(1024, d);

	std::size_t xdim = 16;
	std::size_t ydim = 16;

	std::vector<float> a1(xdim*ydim, 41.);
	std::vector<float> a2(xdim*ydim);

	void * y = opencl_pin(&a1[0], a1.size()*sizeof(float), d);

	module mod = create_module_from_string(kernel_source, d,
			AURA_BACKEND_COMPILE_FLAGS);
	print_module_build_log(mod, d);
	kernel k = create_kernel(mod, "simple_add");

	device_ptr<float> mem = device_malloc<float>(xdim*ydim, d);

	copy(mem, &a1[0], xdim*ydim, f);

	invoke(k, mesh(ydim, xdim), bundle(xdim), args(mem.get()), f);

	copy(&a2[0], mem, xdim*ydim, f);

	wait_for(f);

	for(std::size_t i=0; i<a1.size(); i++) {
		a1[i] += 1.0;
	}
	if(std::equal(a1.begin(), a1.end(), a2.begin())) {
		printf("result seems good!\n");
	}
	device_free(mem);

}


