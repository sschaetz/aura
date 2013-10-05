#ifndef AURA_BACKEND_OPENCL_GRID_HPP
#define AURA_BACKEND_OPENCL_GRID_HPP

#include <array>
#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

typedef svec<std::size_t, AURA_MAX_GRID_DIMS> grid;

} // opencl 
} // backend_detail
} // aura


#endif // AURA_BACKEND_OPENCL_GRID_HPP

