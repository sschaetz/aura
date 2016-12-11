#pragma once

#include <boost/aura/device.hpp>
#include <boost/aura/memory_tag.hpp>
#include <boost/aura/mesh_bundle.hpp>

#include <array>
#include <utility>

namespace boost
{
namespace aura
{

enum mesh_bundle_operation
{
        multiply,
        none,
        divide
};

/// Helper function that adjust mesh and bundle sizes for CUDA and Metal.
/// OpenCL semantics are assumed throughout.
template <typename MeshType, typename BundleType>
std::pair<MeshType, BundleType> adjust_mesh_bundle(
        const MeshType& m, const BundleType& b,
        mesh_bundle_operation op = none)
{
        typename MeshType::value_type meshx = m[0], meshy = 1, meshz = 1;
        typename BundleType::value_type bundlex = b[0], bundley = 1,
                                        bundlez = 1;

        if (m.size() > 1)
        {
                meshy = m[1];
        }
        if (m.size() > 2)
        {
                meshz = m[2];
        }
        if (b.size() > 1)
        {
                bundley = b[1];
        }
        if (b.size() > 2)
        {
                bundlez = b[2];
        }

        switch (op)
        {
                case mesh_bundle_operation::none:
                {
                        break;
                }
                case mesh_bundle_operation::divide:
                {
                        meshx /= bundlex;
                        meshy /= bundley;
                        meshz /= bundlez;
                        break;
                }
                case mesh_bundle_operation::multiply:
                {
                        meshx *= bundlex;
                        meshy *= bundley;
                        meshz *= bundlez;
                        break;
                }
        }

        MeshType ret_mesh;
        ret_mesh[0] = meshx;
        ret_mesh[1] = meshy;
        ret_mesh[2] = meshz;

        BundleType ret_bundle;
        ret_bundle[0] = bundlex;
        ret_bundle[1] = bundley;
        ret_bundle[2] = bundlez;
        return std::make_pair(ret_mesh, ret_bundle);
}

} // aura
} // boost
