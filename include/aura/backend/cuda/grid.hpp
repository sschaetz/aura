#ifndef AURA_BACKEND_CUDA_GRID_HPP
#define AURA_BACKEND_CUDA_GRID_HPP

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

typedef svec<std::size_t, AURA_MAX_GRID_DIMS> grid;

} // cuda
} // backend_detail
} // aura


#endif // AURA_BACKEND_CUDA_GRID_HPP

