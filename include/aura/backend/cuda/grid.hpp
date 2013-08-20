#ifndef AURA_BACKEND_CUDA_GRID_HPP
#define AURA_BACKEND_CUDA_GRID_HPP

#include <array>

namespace aura {
namespace backend_detail {
namespace cuda {

#define AURA_BACKEND_SHARED_GRID_HPP_GUARD
#include <aura/backend/shared/grid.hpp>
#undef AURA_BACKEND_SHARED_GRID_HPP_GUARD

} // cuda
} // backend_detail
} // aura


#endif // AURA_BACKEND_CUDA_GRID_HPP

