#define BOOST_TEST_MODULE backend.factorize

#include <array>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/bounds.hpp>
#include <boost/aura/backend/shared/calc_mesh_bundle.hpp>

// basic_cuda
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic_cuda) 
{
	const std::array<std::size_t, 4> max_mb = {{
		AURA_CUDA_MAX_BUNDLE, 
		AURA_CUDA_MAX_MESH0, 
		AURA_CUDA_MAX_MESH1, 
		AURA_CUDA_MAX_MESH2 
	}};

	std::array<std::size_t, 4> mesh_bundle = {{1, 1, 1, 1}};
	std::array<bool, 4> mask = {{false, false, false, false}};

	
	boost::aura::bounds b(997, 512, 9);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)boost::aura::product(mesh_bundle) == product(b));
	
	mesh_bundle = {{1, 1, 1, 1}};
	b = boost::aura::bounds(1);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)boost::aura::product(mesh_bundle) == product(b));
	
	mesh_bundle = {{1, 1, 1, 1}};
	b = boost::aura::bounds(3, 19, 11);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)boost::aura::product(mesh_bundle) == product(b));

	mesh_bundle = {{1, 1, 1, 1}};
	b = boost::aura::bounds(174, 174);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)boost::aura::product(mesh_bundle) == product(b));
}

// basic_opencl
// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(basic_opencl) 
{

	const std::array<std::size_t, 4> max_mb = {{
		AURA_OPENCL_MAX_BUNDLE, 
		AURA_OPENCL_MAX_MESH0, 
		AURA_OPENCL_MAX_MESH1, 
		AURA_OPENCL_MAX_MESH2 
	}};

	std::array<std::size_t, 4> mesh_bundle = {{1, 1, 1, 1}};
	std::array<bool, 4> mask = {{false, true, false, false}};

	
	boost::aura::bounds b(997, 512, 9);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)(
			boost::aura::product(mesh_bundle)/mesh_bundle[0]) ==
			product(b));

	mesh_bundle = {{1, 1, 1, 1}};
	b = boost::aura::bounds(1);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)boost::aura::product(mesh_bundle) == product(b));
	

	mesh_bundle = {{1, 1, 1, 1}};
	b = boost::aura::bounds(128);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());

	BOOST_CHECK((long int)(
			boost::aura::product(mesh_bundle)/mesh_bundle[0]) ==
			product(b));
	
	mesh_bundle = {{1, 1, 1, 1}};
	b = boost::aura::bounds(3, 19, 11);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)(
			boost::aura::product(mesh_bundle)/mesh_bundle[0]) == 
			product(b));
	
	mesh_bundle = {{1, 1, 1, 1}};
	b = boost::aura::bounds(174, 174, 1);
	boost::aura::detail::calc_mesh_bundle(boost::aura::product(b), 2, 
			mesh_bundle.begin(), max_mb.begin(), mask.begin());
	BOOST_CHECK((long int)(
			boost::aura::product(mesh_bundle)/mesh_bundle[0]) == 
			product(b));

}

