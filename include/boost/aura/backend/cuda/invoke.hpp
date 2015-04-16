#ifndef AURA_BACKEND_CUDA_INVOKE_HPP
#define AURA_BACKEND_CUDA_INVOKE_HPP

#include <cstddef>
#include <cuda.h>
#include <boost/aura/bounds.hpp>
#include <boost/aura/backend/cuda/kernel.hpp>
#include <boost/aura/backend/cuda/call.hpp>
#include <boost/aura/backend/cuda/feed.hpp>
#include <boost/aura/backend/cuda/mesh.hpp>
#include <boost/aura/backend/cuda/bundle.hpp>
#include <boost/aura/backend/cuda/args.hpp>
#include <boost/aura/backend/shared/calc_mesh_bundle.hpp>
#include <boost/config.hpp>

namespace boost {
namespace aura {
namespace backend_detail {
namespace cuda {
namespace detail {


#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template<unsigned long N>
inline void invoke_impl(kernel & k, const mesh & m, const bundle & b,
		const args_t<N>&& a, feed & f)
#else // BOOST_NO_CXX11_VARIADIC_TEMPLATES
inline void invoke_impl(kernel& k, const mesh& m, const bundle& b,
		const args_t& a, feed& f)
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
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

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template<unsigned long N>
inline void invoke_impl(kernel & k, const ::boost::aura::bounds& b,
		const args_t<N>&& a, feed & f)
#else // BOOST_NO_CXX11_VARIADIC_TEMPLATES
inline void invoke_impl(kernel & k, const ::boost::aura::bounds& b,
		const args_t & a, feed & f)
#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
{
	std::array<std::size_t, 4> mb = {{1, 1, 1, 1}};
	const std::array<std::size_t, 4> max_mb = {{
		AURA_CUDA_MAX_BUNDLE,
		AURA_CUDA_MAX_MESH0,
		AURA_CUDA_MAX_MESH1,
		AURA_CUDA_MAX_MESH2
	}};

	std::array<bool, 4> mask = {{false, false, false, false}};

	boost::aura::detail::calc_mesh_bundle(product(b), 2,
			mb.begin(), max_mb.begin(), mask.begin());
	f.set();
	AURA_CUDA_SAFE_CALL(cuLaunchKernel(k, mb[1], mb[2], mb[3],
		mb[0], 1, 1, 0, f.get_backend_stream(),
		const_cast<void**>(&a.second[0]), NULL));
	f.unset();
	free(a.first);
}

} // namespace detail

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES

/// invoke kernel without args
inline void invoke(kernel& k, const mesh& m, const bundle& b, feed& f)
{
	detail::invoke_impl(k, m, b, args_t<0>(), f);
}

/// invoke kernel with args
template<unsigned long N>
inline void invoke(kernel& k, const mesh& m, const bundle& b,
		const args_t<N>&& a, feed& f)
{
	detail::invoke_impl(k, m, b, std::move(a), f);
}

/// invoke kernel with bounds and args
template<unsigned long N>
inline void invoke(kernel& k, const bounds& b, const args_t<N>&& a, feed& f)
{
	detail::invoke_impl(k, b, std::move(a), f);
}

/// invoke kernel with size and args
template<unsigned long N>
inline void invoke(kernel& k, const std::size_t s, const args_t<N>&& a, feed& f)
{
	detail::invoke_impl(k, bounds(s), std::move(a), f);
}

#else // BOOST_NO_CXX11_VARIADIC_TEMPLATES

/// invoke kernel without args
inline void invoke(kernel& k, const mesh& m, const bundle& b, feed& f)
{
	detail::invoke_impl(k, m, b, args_t(), f);
}

/// invoke kernel with args
inline void invoke(kernel& k, const mesh& m, const bundle& b,
		const args_t& a, feed& f)
{
	detail::invoke_impl(k, m, b, a, f);
}

/// invoke kernel with bounds and args
inline void invoke(kernel& k, const bounds& b, const args_t& a, feed& f)
{
	detail::invoke_impl(k, b, a, f);
}

/// invoke kernel with size and args
inline void invoke(kernel& k, const std::size_t s, const args_t& a, feed& f)
{
	detail::invoke_impl(k, bounds(s), a, f);
}

#endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES


} // namespace cuda
} // namespace backend_detail
} // namespace aura
} // namespace boost

#endif // AURA_BACKEND_CUDA_INVOKE_HPP

