#ifndef AURA_BACKEND_CUDA_INVOKE_HPP
#define AURA_BACKEND_CUDA_INVOKE_HPP

#include <cstddef>
#include <cuda.h>
#include <aura/backend/cuda/kernel.hpp>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/feed.hpp>
#include <aura/backend/cuda/mesh.hpp>
#include <aura/backend/cuda/bundle.hpp>
#include <aura/backend/cuda/args.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

namespace detail {

void invoke_impl(kernel & k, const mesh & m, const bundle & b, 
		const args_t & a, feed & f) 
{
	// handling for non 3-dimensional mesh and bundle sizes
	std::size_t meshx = m[0], meshy = 1, meshz = 1;
	std::size_t bundlex = b[0], bundley = 1, bundlez = 1;
	
	if (m.size() > 1) {
		meshy = m[1];
	}
	if (m.size() > 2) {
		meshz = m[2];
	}
	if (b.size() > 1) {
		bundley = b[1];
	}
	if (b.size() > 2) {
		bundlez = b[2];
	}

	// number of bundles subdivides meshes but CUDA has a
	// "consists of" semantic so we need less mesh elements
	meshx /= bundlex;
	meshy /= bundley;
	meshz /= bundlez;

	f.set();
	AURA_CUDA_SAFE_CALL(cuLaunchKernel(k, meshx, meshy, meshz, 
		bundlex, bundley, bundlez, 0, f.get_backend_stream(), 
		const_cast<void**>(&a.second[0]), NULL)); 
	f.unset();
	free(a.first);
}

/**
 * @brief calculate combination of bundle and mesh, based on v
 *
 * recursively calculates the integer factorization and fills 
 * up an array of mesh and bundle (iterator i), rules are 
 * defined by a maximum size for mesh and bundle (iterator b)
 */
void calc_mesh_bundle(std::size_t v, std::size_t f, 
		std::array<std::size_t, 4>::iterator i,
		std::array<std::size_t, 4>::const_iterator b)
{
	if(f > v) {
		return;
	}
	if (0 == v % f) {
		if(*i*f > *b) {
			++i;
			++b;
			// if next dimension can hold value
			// the size is invalid
			assert(*i*f < *b);
		}
		// put new factor in	
		*i *= f;
		calc_mesh_bundle(v/f, f, i, b);
	} else {
		f++;
		calc_mesh_bundle(v, f, i, b);	
	}
}

void invoke_impl(kernel & k, const bounds& b, const args_t & a, feed & f) 
{
	std::array<std::size_t, 4> mb = {{1, 1, 1, 1}};
	const std::array<std::size_t, 4> max_mb = {{
		AURA_CUDA_MAX_BUNDLE, 
		AURA_CUDA_MAX_MESH0, 
		AURA_CUDA_MAX_MESH1, 
		AURA_CUDA_MAX_MESH2 
	}};
	
	calc_mesh_bundle(product(b), 2, 
			mb.begin(), max_mb.begin());
	f.set();
	AURA_CUDA_SAFE_CALL(cuLaunchKernel(k, mb[1], mb[2], mb[3], 
		mb[0], 1, 1, 0, f.get_backend_stream(), 
		const_cast<void**>(&a.second[0]), NULL)); 
	f.unset();
	free(a.first);
}

} // namespace detail

/// invoke kernel without args
void invoke(kernel& k, const mesh& m, const bundle& b, feed& f) 
{
	detail::invoke_impl(k, m, b, args_t(), f);
}

/// invoke kernel with args
void invoke(kernel& k, const mesh& m, const bundle& b, 
		const args_t& a, feed& f) 
{
	detail::invoke_impl(k, m, b, a, f);
}

/// invoke kernel with bounds and args
void invoke(kernel& k, const bounds& b, const args_t& a, feed& f) 
{
	detail::invoke_impl(k, b, a, f);
}

} // namespace aura
} // namespace backend_detail
} // namespace cuda

#endif // AURA_BACKEND_CUDA_INVOKE_HPP

