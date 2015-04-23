#ifndef AURA_MATH_PARTITION_MESH_HPP
#define AURA_MATH_PARTITION_MESH_HPP

#include <tuple>

#include <stdio.h>
#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>

namespace boost
{
namespace aura
{
namespace math
{

void partition_mesh(std::vector<std::size_t> & mesh_size, 
		std::size_t numel, std::size_t bundle_size)
{
	// multiple of BUNDLE_SIZE in entire Volume
	std::size_t numbs = (numel/bundle_size+1)*bundle_size;		
	
	// partition the mesh in multiples of BUNDLE_SIZE 
	// while activating only as many fibers as needed
	if ((float)numbs / (float)(AURA_OPENCL_MAX_MESH0 * 
				AURA_OPENCL_MAX_MESH1) <= 1) {
	
		mesh_size[2] = 1;
		
		// everything fits into one dimension	
		if ((float)numbs / (float)(AURA_OPENCL_MAX_MESH0) <= 1){	
			mesh_size[1] = 1;			
			mesh_size[0] = numbs;
		} else {	
			// we need two dimensions
			mesh_size[0] = ceil( (float)numel / 
					(float)(AURA_OPENCL_MAX_MESH1 * 
						bundle_size) ) * 
				bundle_size;
			mesh_size[1] = ceil( (float)numel / 
					(float)(mesh_size[0]));
			// in case of rounding errors
			if (mesh_size[0]*mesh_size[1] < numel ) {
				mesh_size[0] = (numel /
						(AURA_OPENCL_MAX_MESH1 *
						 bundle_size)+1) * bundle_size;
				mesh_size[1] = ( numel / (mesh_size[0]) +1);
			}
		}
	} else { 	
		// we need three dimensions
		if((float)numbs / (float)(AURA_OPENCL_MAX_MESH0 * 
					AURA_OPENCL_MAX_MESH1 * 
					AURA_OPENCL_MAX_MESH2) > 1) {
			printf("Input array is too large. Aborting...");
			return;
		}
		mesh_size[0] = ceil( (float)numel / 
				(float)(AURA_OPENCL_MAX_MESH1 * 
					AURA_OPENCL_MAX_MESH2 * 
					bundle_size)  ) * bundle_size;				
		mesh_size[1] = ceil( (float)numel / 
				(float)(mesh_size[0] * 
					AURA_OPENCL_MAX_MESH2) );
		mesh_size[2] = ceil( (float)numel / 
				(float)(mesh_size[0] * mesh_size[1]) );
		// in case of rounding errors
		if (mesh_size[0]*mesh_size[1]*mesh_size[2] < numel ) {
			mesh_size[0] = ( numel / 
					(AURA_OPENCL_MAX_MESH1 * 
					 AURA_OPENCL_MAX_MESH2 * 
					 bundle_size)+1) *
				bundle_size;				
			mesh_size[1] = ( numel / (mesh_size[0] * 
						AURA_OPENCL_MAX_MESH2) +1);
			mesh_size[2] = ( numel / 
					(mesh_size[0] * mesh_size[1]) +1);
		}
	}
}


} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_PARTITION_MESH_HPP

