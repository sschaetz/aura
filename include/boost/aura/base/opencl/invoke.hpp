#pragma once

#include <boost/aura/base/base_mesh_bundle.hpp>
#include <boost/aura/base/opencl/feed.hpp>
#include <boost/aura/base/opencl/kernel.hpp>
#include <boost/aura/base/opencl/safecall.hpp>
#include <boost/aura/meta/tsizeof.hpp>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
{

typedef std::pair<void*, std::size_t> arg_t;
template <std::size_t N>
using args_tt = std::array<arg_t, N>;

// alias for returned packed arguments
template <std::size_t N>
using args_t = std::pair<char*, args_tt<N>>;

/// Copy arguments to memory block recursively
template <typename ArgsItr, typename T0>
void fill_args_(char* p, ArgsItr it, const T0 a0)
{
        std::memcpy(p, &a0, sizeof(T0));
        *it = std::make_pair(p, sizeof(T0));
}

template <typename ArgsItr, typename T0, typename... Targs>
void fill_args_(char* p, ArgsItr it, const T0 a0, const Targs... ar)
{
        std::memcpy(p, &a0, sizeof(T0));
        *it = std::make_pair(p, sizeof(T0));
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

template <unsigned long N>
inline void invoke_impl(
        kernel& k, const mesh& m, const bundle& b, const args_t<N>&& a, feed& f)
{
        // set parameters
        for (std::size_t i = 0; i < a.second.size(); i++)
        {
                AURA_OPENCL_SAFE_CALL(clSetKernelArg(k.get_base_kernel(), i,
                        a.second[i].second, a.second[i].first));
        }

        auto mesh_bundle = adjust_mesh_bundle(m, b);

        // call kernel
        AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(f.get_base_feed(),
                k.get_base_kernel(), mesh_bundle.first.size(), NULL,
                &mesh_bundle.first[0], &mesh_bundle.second[0], 0, NULL, NULL));
        free(a.first);
}

} // namespace detail



/// invoke kernel without args
template <typename MeshType, typename BundleType>
inline void invoke(kernel& k, const MeshType& m, const BundleType& b, feed& f)
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


} // opencl
} // base_detail
} // aura
} // boost
