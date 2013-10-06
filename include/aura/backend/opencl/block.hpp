#ifndef AURA_BACKEND_OPENCL_BLOCK_HPP
#define AURA_BACKEND_OPENCL_BLOCK_HPP

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

typedef svec<std::size_t, AURA_MAX_BLOCK_DIMS> block;

} // opencl 
} // backend_detail
} // aura


#endif // AURA_BACKEND_OPENCL_BLOCK_HPP

