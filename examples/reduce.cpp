#include <numeric>

#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

int main(void) {
	initialize();
	device d(0);  
	feed f(d);
	std::size_t xdim = 32;
	std::size_t ydim = 32;
	bounds b(xdim, ydim);
	
	std::vector<float> a1(xdim*ydim);
	{
		int cur = 0;
		std::generate(a1.begin(), a1.end(), 
				[&]() { return (float)cur++; } );
	}
	float r = 0.0;	

	module mod = create_module_from_file("reduce.cc", d, 
		AURA_BACKEND_COMPILE_FLAGS);

	print_module_build_log(mod, d);
	kernel k = create_kernel(mod, "red1");
	
	device_array<float> mem(b, d);
	device_array<float> rd(1, d);

	copy(a1, mem, f);
	copy(&r, rd, f);

	invoke(k, mesh(xdim, ydim), bundle(16), 
			args(mem.begin_ptr(), rd.begin_ptr()), f);
	
	copy(rd, &r, f);
	copy(mem, a1, f);
	wait_for(f);
	
	for(auto x : a1) {
		std::cout << x << " ";
	}

	std::cout << std::endl;
	std::cout << r << std::endl;
	std::cout << std::accumulate(a1.begin(), a1.end(), 0.0) << std::endl;
	std::cout << std::endl << "done." << std::endl;

}

