
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

int main(void)
{
	initialize();
	int num = device_get_count();

	AURA_CHECK_ERROR(0 < num)
	device d(0);
	feed f(d);

	std::size_t xdim = 16;
	std::size_t ydim = 16;

	std::vector<float> a1(xdim*ydim, 41.);
	std::vector<float> a2(xdim*ydim);

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


