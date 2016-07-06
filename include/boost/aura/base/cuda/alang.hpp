#pragma once

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{

struct alang_header
{
        static const std::string& get()
        {
                static std::string v = R"(

#define AURA_KERNEL extern "C" __global__
#define AURA_CONSTANT
#define AURA_DEVMEM

#define AURA_MESH_ID_ARG
#define AURA_MESH_ID_0 (blockIdx.x * blockDim.x + threadIdx.x)
#define AURA_MESH_ID_1 (blockIdx.y * blockDim.y + threadIdx.y)
#define AURA_MESH_ID_2 (blockIdx.z * blockDim.z + threadIdx.z)

#define AURA_MESH_SIZE_ARG
#define AURA_MESH_SIZE_0 (gridDim.x * blockDim.x)
#define AURA_MESH_SIZE_1 (gridDim.y * blockDim.y)
#define AURA_MESH_SIZE_2 (gridDim.z * blockDim.z)

#define AURA_BUNDLE_ID_ARG
#define AURA_BUNDLE_ID_0 threadIdx.x
#define AURA_BUNDLE_ID_1 threadIdx.y
#define AURA_BUNDLE_ID_2 threadIdx.z

#define AURA_BUNDLE_SIZE_ARG
#define AURA_BUNDLE_SIZE_0 blockDim.x
#define AURA_BUNDLE_SIZE_1 blockDim.y
#define AURA_BUNDLE_SIZE_2 blockDim.z

)";
                return v;
        }
};

} // cuda
} // base_detail
} // namespace aura
} // namespace boost
