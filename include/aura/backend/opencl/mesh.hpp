#ifndef AURA_BACKEND_OPENCL_MESH_HPP
#define AURA_BACKEND_OPENCL_MESH_HPP

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

typedef svec<std::size_t, AURA_MAX_MESH_DIMS> mesh;

} // opencl 
} // backend_detail
} // aura


#endif // AURA_BACKEND_OPENCL_MESH_HPP

