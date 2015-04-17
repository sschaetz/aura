#define BOOST_TEST_MODULE math.dot

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/blas/dot.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/math/complex.hpp>
#include <boost/aura/math/partition_mesh.hpp>

using namespace boost::aura;

// partition_mesh
// _____________________________________________________________________________
 
BOOST_AUTO_TEST_CASE(partition_mesh) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,8,13,16,24,28,34,80,128,1011,
		1024,1031,1024*256-1,1024*256,1024*256+1,1024*512-1,1024*512,
		1024*512+1,1024*1024-1,1024*1024-3,1024*1024+1,1024*1024*2-1,
		1024*1024*4-3,1024*1024*8-7,1024*1024*16-531,1024*1024*512-1,
		1024*1024*512,1024*1024*512+1};
	std::vector<int> bundles = {1,2,4,8,16,32,64,128,256,512,1024,
		3,5,11,13,113,513};

	for (auto x : sizes) {
		for (auto b : bundles) {
			std::vector<std::size_t> mesh_size(3,1);
			
			math::partition_mesh(mesh_size,x,b);
			
			// debug output (mesh size and mesh overhead)
			std::cout << "bundle_size:" << b << std::endl;
			std::cout << "mesh_size = " << mesh_size[0] << 
				" " << mesh_size[1] << " " << mesh_size[2] << 
				std::endl << " total necessary " << x << 
				std::endl << " total activated " << 
				mesh_size[0]*mesh_size[1]*mesh_size[2] << 
				std::endl;
			
			BOOST_CHECK(mesh_size[0]*mesh_size[1]*mesh_size[2] 
					>= x);
			BOOST_CHECK(mesh_size[0]*mesh_size[1]*mesh_size[2] <= 
					AURA_OPENCL_MAX_MESH0*
					AURA_OPENCL_MAX_MESH1*
					AURA_OPENCL_MAX_MESH2
				);
		}
	}
}

