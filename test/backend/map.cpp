#define BOOST_TEST_MODULE backend.map

#include <cstring>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>
#include <aura/device_view.hpp>

using namespace aura;
using namespace aura::backend;

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
	
	device_view<float> src_view = map<float>(src, d);
	BOOST_CHECK(&src[0] == nullptr);
	
	unmap(src_view, src, f);
	wait_for(f);
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
	device_ptr<float> src_mapped = device_map(&src[0], src.size(), d);
	device_ptr<float> dst_mapped = device_map(&dst[0], dst.size(), d);
	
	invoke(k, mesh(1024), bundle(1024/16), 
			args(dst_mapped.get(), src_mapped.get()), f);
	
	wait_for(f);
	
	// CUDA would be done here
	
	// OpenCL needs to sync the data back to host memory
	device_unmap(&src[0], src_mapped, src.size(), f);
	device_unmap(&dst[0], dst_mapped, src.size(), f);
	
	wait_for(f);
	BOOST_CHECK(std::equal(dst.begin(), dst.begin()+dst.size()/2, 
				cmp.begin()));
	BOOST_CHECK(std::count(dst.begin()+dst.size()/2, 
				dst.end(), 0.) == (signed)dst.size()/2);
	BOOST_CHECK(src.size() == 1024);
	BOOST_CHECK(dst.size() == 1024*2);
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

	
	device_view<float> src_view = map<float>(src, d);
	device_view<float> dst_view = map<float>(dst, d);
	
	
	BOOST_CHECK(src.empty());
	BOOST_CHECK(dst.empty());

	feed f(d);
	invoke(k, mesh(1024), bundle(1024/16), 
			args(dst_view.begin_ptr(), src_view.begin_ptr()), f);
	
	unmap(src_view, src, f);
	unmap(dst_view, dst, f);

	
	wait_for(f);

	BOOST_CHECK(std::equal(dst.begin(), dst.begin()+dst.size()/2, 
				cmp.begin()));
	BOOST_CHECK(std::count(dst.begin()+dst.size()/2, 
				dst.end(), 0.) == (signed)dst.size()/2);

	BOOST_CHECK(src.size() == 1024);
	BOOST_CHECK(dst.size() == 1024*2);
	BOOST_CHECK(&src[0] == src_buf);
	BOOST_CHECK(&dst[0] == dst_buf);
}


