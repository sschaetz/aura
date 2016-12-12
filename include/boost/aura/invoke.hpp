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

/// Defines if mesh defines all threads or only the mesh size.
enum mesh_definition
{
        all_threads,
        mesh_size,
};

/// Normalize mesh to the aura standard definition of mesh:
/// mesh defines the number of all threads that are launched
/// on the accelerator.
template <typename MeshType, typename BundleType>
MeshType normalize_mesh(MeshType mesh, BundleType bundle,
        mesh_definition mesh_def)
{
        switch (mesh_def)
        {
                case mesh_definition::all_threads:
                {
                        break;
                }
                case mesh_definition::mesh_size:
                {
                        for (std::size_t i = 0; i<mesh.size(); i++)
                        {
                                mesh[i] *= bundle[i];
                        }
                }
        }
        return mesh;
}

/// Pack arguments
template <typename... Targs>
auto args(const Targs... ar) -> base::args_t<sizeof...(Targs)>
{
        return base::args_impl(ar...);
}

/// invoke kernel without args
template <typename MeshType, typename BundleType>
inline void invoke(kernel& k, const MeshType& m, const BundleType& b, feed& f,
        mesh_definition mesh_def = mesh_definition::mesh_size)
{
        auto normalized_mesh = normalize_mesh(m, b, mesh_def);
        base::detail::invoke_impl(k, normalize_mesh, b, base::args_t<0>(), f);
}

/// invoke kernel with args
template <unsigned long N, typename MeshType, typename BundleType>
inline void invoke(kernel& k, const MeshType& m, const BundleType& b,
        const base::args_t<N>&& a, feed& f,
        mesh_definition mesh_def = mesh_definition::mesh_size)
{
        auto normalized_mesh = normalize_mesh(m, b, mesh_def);
        base::detail::invoke_impl(k, normalized_mesh, b, std::move(a), f);
}


} // namespace aura
} // namespace boost
