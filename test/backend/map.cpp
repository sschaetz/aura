#define BOOST_TEST_MODULE backend.map

#include <cstring>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/device_map.hpp>

using namespace boost::aura;
using namespace boost::aura::backend;

const char * kernel_file = AURA_UNIT_TEST_LOCATION"map.cc";

// moving 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(moving) 
{
	initialize();
	int num = device_get_count();
	if(num < 1) {
		return;	
	}    
	
	device d(0); 
	feed f(d);
	std::vector<float> src(1024, 42.);
	
	BOOST_CHECK(&src[0] != nullptr);
	float* tmp = &src[0];
	
	device_map<float> src_map(src, memory_tag::rw, d);
	BOOST_CHECK(&src[0] == nullptr);
	
	src_map.unmap(src, f);
	
	BOOST_CHECK(&src[0] == tmp);
}

// backend
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(lowlevel) 
{
	initialize();
	int num = device_get_count();
	if(num < 1) {
		return;	
	}    
	
	device d(0); 
	module m = create_module_from_file(kernel_file, d, 
			AURA_BACKEND_COMPILE_FLAGS);
	kernel k = create_kernel(m, "copy");

	std::vector<float> src(1024, 42.);
	std::vector<float> dst(1024*2, 0.);
	std::vector<float> cmp(1024, 42.);

	feed f(d);
	device_ptr<float> src_mapped = device_map_alloc(&src[0], src.size(), 
			memory_tag::ro, d);
	device_ptr<float> dst_mapped = device_map_alloc(&dst[0], dst.size(), 
			memory_tag::wo, d);
	
	invoke(k, mesh(1024), bundle(1024/16), 
			args(dst_mapped.get(), src_mapped.get()), f);
	
	// OpenCL needs to sync the data back to host memory
	device_unmap(&dst[0], dst_mapped, src.size(), f);
	
	BOOST_CHECK(std::equal(dst.begin(), dst.begin()+dst.size()/2, 
				cmp.begin()));
	BOOST_CHECK(std::count(dst.begin()+dst.size()/2, 
				dst.end(), 0.) == (signed)dst.size()/2);
}

// frontend 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(frontend) 
{
	initialize();
	int num = device_get_count();
	if(num < 1) {
		return;	
	}    
	
	device d(0); 
	module m = create_module_from_file(kernel_file, d, 
			AURA_BACKEND_COMPILE_FLAGS);
	kernel k = create_kernel(m, "copy");

	std::vector<float> src(1024, 42.);
	std::vector<float> dst(1024*2, 0.);
	std::vector<float> cmp(1024, 42.);

	float* src_buf = &src[0];
	float* dst_buf = &dst[0];

	
	device_map<float> src_map(src, memory_tag::ro, d);
	device_map<float> dst_map(dst, memory_tag::wo, d);
	
	
	BOOST_CHECK(src.empty());
	BOOST_CHECK(dst.empty());

	feed f(d);
	invoke(k, mesh(1024), bundle(1024/16), 
			args(dst_map.begin_ptr(), src_map.begin_ptr()), f);
	
	src_map.unmap(src, f);	
	dst_map.unmap(dst, f);	

	BOOST_CHECK(std::equal(dst.begin(), dst.begin()+dst.size()/2, 
				cmp.begin()));
	BOOST_CHECK(std::count(dst.begin()+dst.size()/2, 
				dst.end(), 0.) == (signed)dst.size()/2);

	BOOST_CHECK(src.size() == 1024);
	BOOST_CHECK(dst.size() == 1024*2);
	BOOST_CHECK(&src[0] == src_buf);
	BOOST_CHECK(&dst[0] == dst_buf);
}

