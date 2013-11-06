#ifndef AURA_BACKEND_CUDA_MESH_HPP
#define AURA_BACKEND_CUDA_MESH_HPP

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

typedef svec<std::size_t, AURA_MAX_MESH_DIMS> mesh;

} // cuda
} // backend_detail
} // aura


#endif // AURA_BACKEND_CUDA_MESH_HPP

