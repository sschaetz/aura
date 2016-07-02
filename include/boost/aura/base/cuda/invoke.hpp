#pragma once

#include <boost/aura/base/base_mesh_bundle.hpp>
#include <boost/aura/base/cuda/feed.hpp>
#include <boost/aura/base/cuda/kernel.hpp>
#include <boost/aura/base/cuda/safecall.hpp>
#include <boost/aura/meta/tsizeof.hpp>

#include <cuda.h>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{

typedef void* arg_t;
template <std::size_t N>
using args_tt = std::array<arg_t, N>;

/// Alias for returned packed arguments
template <std::size_t N>
using args_t = std::pair<char*, args_tt<N>>;

/// Unwrap argument.
template <typename T>
const T* unwrap(const T& a)
{
        return &a;
}

/// Copy arguments to memory block recursively
template <typename ArgsItr, typename T0>
void fill_args_(char* p, ArgsItr it, const T0 a0)
{
        std::memcpy(p, unwrap(a0), sizeof(T0));
        *it = p;
}

template <typename ArgsItr, typename T0, typename... Targs>
void fill_args_(char* p, ArgsItr it, const T0 a0, const Targs... ar)
{
        std::memcpy(p, unwrap(a0), sizeof(T0));
        *it = p;
        fill_args_(p + sizeof(T0), ++it, ar...);
}

/// Pack arguments
template <typename... Targs>
args_t<sizeof...(Targs)> args(const Targs... ar)
{
        args_tt<sizeof...(Targs)> pa;
        char* p = (char*)malloc(tsizeof<Targs...>::sz);
        char* ptr = p;
        fill_args_(p, pa.begin(), ar...);
        return std::make_pair(ptr, pa);
}

namespace detail
{

template <unsigned long N, typename MeshType, typename BundleType>
inline void invoke_impl(kernel& k, const MeshType& m, const BundleType& b,
        const args_t<N>&& a, feed& f)
{
        auto mesh_bundle = adjust_mesh_bundle(m, b);
        f.get_device().activate();

        AURA_CUDA_SAFE_CALL(cuLaunchKernel(k.get_base_kernel(),
                mesh_bundle.first[0], mesh_bundle.first[1],
                mesh_bundle.first[2], mesh_bundle.second[0],
                mesh_bundle.second[1], mesh_bundle.second[2], 0,
                f.get_base_feed(), const_cast<void**>(&a.second[0]), NULL));
        f.get_device().deactivate();
        free(a.first);
}


} // namespace detail

/// invoke kernel without args
inline template <typename MeshType, typename BundleType>
void invoke(kernel& k, const MeshType& m, const BundleType& b, feed& f)
{
        detail::invoke_impl(k, m, b, args_t<0>(), f);
}

/// invoke kernel with args
template <unsigned long N, typename MeshType, typename BundleType>
inline void invoke(kernel& k, const MeshType& m, const BundleType& b,
        const args_t<N>&& a, feed& f)
{
        detail::invoke_impl(k, m, b, std::move(a), f);
}

} // cuda
} // base_detail
} // aura
} // boost
