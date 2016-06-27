#pragma once

#include <boost/aura/base/base_mesh_bundle.hpp>
#include <boost/aura/base/metal/feed.hpp>
#include <boost/aura/base/metal/kernel.hpp>
#include <boost/aura/base/metal/safecall.hpp>
#include <boost/aura/meta/tsizeof.hpp>

#import <Metal/Metal.h>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace metal
{

typedef id<MTLBuffer> arg_t;
template <unsigned long N>
using args_tt = std::array<arg_t, N>;

// Alias for returned packed arguments.
template <int N>
using args_t = args_tt<N>;

/// Copy arguments to memory block recursively.
template <typename T0>
void fill_args_(args_tt<0>::iterator it, const T0 a0)
{
        *it = a0.device_buffer;
}

template <typename T0, typename... Targs>
void fill_args_(args_tt<0>::iterator it, const T0 a0, const Targs... ar)
{
        *it = a0.device_buffer;
        fill_args_(++it, ar...);
}

/// Pack arguments.
template <typename... Targs>
args_t<sizeof...(Targs)> args(const Targs... ar)
{
        args_tt<sizeof...(Targs)> pa;
        fill_args_(pa.begin(), ar...);
        return pa;
}

namespace detail
{

template <unsigned long N, typename MeshType, typename BundleType>
inline void invoke_impl(kernel& k, const MeshType& m, const BundleType& b,
        const args_t<N>&& a, feed& f)
{
        auto mesh_bundle = adjust_mesh_bundle(m, b);
        command_buffer& cmdb = f.get_command_buffer();
        AURA_METAL_CHECK_ERROR(cmdb.command_buffer);

        id<MTLComputePipelineState> pstate =
                [f.get_device().get_base_device()
                        newComputePipelineStateWithFunction:k.get_base_kernel()
                                                      error:nil];

        AURA_METAL_CHECK_ERROR(pstate);

        id<MTLComputeCommandEncoder> enc =
                [cmdb.command_buffer computeCommandEncoder];

        AURA_METAL_CHECK_ERROR(enc);

        [enc setComputePipelineState:pstate];

        // Set parameters.
        for (std::size_t i = 0; i < a.size(); i++)
        {
                [enc setBuffer:a[i] offset:0 atIndex:i];
        }

        std::cout << mesh_bundle.first[0] << " " << mesh_bundle.first[1] << " "
                  << mesh_bundle.first[2] << " " << mesh_bundle.second[0] << " "
                  << mesh_bundle.second[1] << " " << mesh_bundle.second[2]
                  << std::endl;

        MTLSize threadGroups = MTLSizeMake(mesh_bundle.first[0],
                mesh_bundle.first[1], mesh_bundle.first[2]);
        MTLSize threadsPerGroup = MTLSizeMake(mesh_bundle.second[0],
                mesh_bundle.second[1], mesh_bundle.second[2]);
        [enc dispatchThreadgroups:threadGroups
                threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cmdb.command_buffer commit];
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

} // metal
} // base_detail
} // aura
} // boost
