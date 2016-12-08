#pragma once

#include <boost/aura/base/base_mesh_bundle.hpp>
#include <boost/aura/kernel.hpp>

#if defined AURA_BASE_CUDA
#include <boost/aura/base/cuda/invoke.hpp>
#elif defined AURA_BASE_OPENCL
#include <boost/aura/base/opencl/invoke.hpp>
#elif defined AURA_BASE_METAL
#include <boost/aura/base/metal/invoke.hpp>
#endif

namespace boost
{
namespace aura
{

#if defined AURA_BASE_CUDA
namespace base = base_detail::cuda;
#elif defined AURA_BASE_OPENCL
namespace base = base_detail::opencl;
#elif defined AURA_BASE_METAL
namespace base = base_detail::metal;
#endif



/// Pack arguments
template <typename... Targs>
auto args(const Targs... ar) -> base::args_t<sizeof...(Targs)>
{
        return base::args_impl(ar...);
}

/// invoke kernel without args
template <typename MeshType, typename BundleType>
inline void invoke(kernel& k, const MeshType& m, const BundleType& b, feed& f)
{
        base::detail::invoke_impl(k, m, b, base::args_t<0>(), f);
}

/// invoke kernel with args
template <unsigned long N, typename MeshType, typename BundleType>
inline void invoke(kernel& k, const MeshType& m, const BundleType& b,
        const base::args_t<N>&& a, feed& f)
{
        base::detail::invoke_impl(k, m, b, std::move(a), f);
}


} // namespace aura
} // namespace boost
