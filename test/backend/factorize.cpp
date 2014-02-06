#define BOOST_TEST_MODULE backend.factorize

#include <array>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {

#ifdef AURA_BACKEND_CUDA
	const std::array<std::size_t, 4> max_mb = {{
		AURA_CUDA_MAX_BUNDLE, 
		AURA_CUDA_MAX_MESH0, 
		AURA_CUDA_MAX_MESH1, 
		AURA_CUDA_MAX_MESH2 
	}};
#endif

	std::array<std::size_t, 4> mesh_bundle = {{1, 1, 1, 1}};

	
	aura::bounds b(997, 512, 9);
	aura::backend::detail::calc_mesh_bundle(aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin());
	BOOST_CHECK((long int)aura::product(mesh_bundle) == product(b));
	
	mesh_bundle = {{1, 1, 1, 1}};
	b = aura::bounds(1);
	aura::backend::detail::calc_mesh_bundle(aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin());
	BOOST_CHECK((long int)aura::product(mesh_bundle) == product(b));
	
	mesh_bundle = {{1, 1, 1, 1}};
	b = aura::bounds(3, 19, 11);
	aura::backend::detail::calc_mesh_bundle(aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin());
	BOOST_CHECK((long int)aura::product(mesh_bundle) == product(b));
}

