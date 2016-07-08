#pragma once

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace metal
{

struct alang_header
{
        static const std::string& get()
        {
                static std::string v = R"(

// PYTHON-BEGIN

#include <metal_stdlib>
#include <metal_compute>
#include <metal_math>

using namespace metal;

#define AURA_KERNEL kernel
#define AURA_CONSTANT __constant
#define AURA_DEVMEM device

#define AURA_MESH_ID_ARG , uint3 aura_mesh_id[[thread_position_in_grid]]
#define AURA_MESH_ID_0 aura_mesh_id.x
#define AURA_MESH_ID_1 aura_mesh_id.y
#define AURA_MESH_ID_2 aura_mesh_id.z

#define AURA_MESH_SIZE_ARG , uint3 aura_mesh_size[[threads_per_grid]]
#define AURA_MESH_SIZE_0 aura_mesh_size.x
#define AURA_MESH_SIZE_1 aura_mesh_size.y
#define AURA_MESH_SIZE_2 aura_mesh_size.z

#define AURA_BUNDLE_ID_ARG \
        , uint3 aura_bundle_id[[thread_position_in_threadgroup]]
#define AURA_BUNDLE_ID_0 aura_bundle_id.x
#define AURA_BUNDLE_ID_1 aura_bundle_id.y
#define AURA_BUNDLE_ID_2 aura_bundle_id.z

#define AURA_BUNDLE_SIZE_ARG , uint3 aura_bundle_size[[threads_per_threadgroup]]
#define AURA_BUNDLE_SIZE_0 aura_bundle_size.x
#define AURA_BUNDLE_SIZE_1 aura_bundle_size.y
#define AURA_BUNDLE_SIZE_2 aura_bundle_size.z

// PYTHON-END

)";
                return v;
        }
};

} // metal
} // base_detail
} // aura
} // boost
