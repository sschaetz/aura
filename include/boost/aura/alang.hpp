#pragma once

namespace boost
{
namespace aura
{

const char* alang_header_global = R"(
#define M_PI 3.14159265358979323846
)";

const char* alang_header_opencl = R"(
#define AURA_KERNEL __kernel
#define AURA_CONSTANT __constant
#define AURA_DEVMEM __global

#define AURA_MESH_ID_ARG
#define AURA_MESH_ID_0 get_global_id(0)
#define AURA_MESH_ID_1 get_global_id(1)
#define AURA_MESH_ID_2 get_global_id(2)

#define AURA_MESH_SIZE_ARG
#define AURA_MESH_SIZE_0 get_global_size(0)
#define AURA_MESH_SIZE_1 get_global_size(1)
#define AURA_MESH_SIZE_2 get_global_size(2)

#define AURA_BUNDLE_ID_ARG
#define AURA_BUNDLE_ID_0 get_local_id(0)
#define AURA_BUNDLE_ID_1 get_local_id(1)
#define AURA_BUNDLE_ID_2 get_local_id(2)

#define AURA_BUNDLE_SIZE_ARG
#define AURA_BUNDLE_SIZE_0 get_local_size(0)
#define AURA_BUNDLE_SIZE_1 get_local_size(1)
#define AURA_BUNDLE_SIZE_2 get_local_size(2)
)";

const char* alang_header_metal = R"(
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
)";


const char* alang_header_cuda = R"(
#define AURA_KERNEL extern "C" __global__
#define AURA_CONSTANT
#define AURA_DEVMEM

#define AURA_MESH_ID_ARG
#define AURA_MESH_ID_0 blockIdx.x * blockDim.x + threadIdx.x
#define AURA_MESH_ID_1 blockIdx.y * blockDim.y + threadIdx.y
#define AURA_MESH_ID_2 blockIdx.z * blockDim.z + threadIdx.z

#define AURA_MESH_SIZE_ARG
#define AURA_MESH_SIZE_0 gridDim.x * blockDim.x
#define AURA_MESH_SIZE_1 gridDim.y * blockDim.y
#define AURA_MESH_SIZE_2 gridDim.z * blockDim.z

#define AURA_BUNDLE_ID_ARG
#define AURA_BUNDLE_ID_0 threadId.x
#define AURA_BUNDLE_ID_1 threadId.y
#define AURA_BUNDLE_ID_2 threadId.z

#define AURA_BUNDLE_SIZE_ARG
#define AURA_BUNDLE_SIZE_0 blockDim.x
#define AURA_BUNDLE_SIZE_1 blockDim.y
#define AURA_BUNDLE_SIZE_2 blockDim.z
)";

} // namespace aura
} // namespace boost
