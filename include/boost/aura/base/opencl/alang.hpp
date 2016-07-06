#pragma once

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
{

struct alang_header
{
        static const std::string& get()
        {
                static std::string v = R"(

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
                return v;
        }
};


} // opencl
} // base_detail
} // aura
} // boost
