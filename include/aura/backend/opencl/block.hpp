#ifndef AURA_BACKEND_OPENCL_BLOCK_HPP
#define AURA_BACKEND_OPENCL_BLOCK_HPP


#include <array>

namespace aura {
namespace backend_detail {
namespace opencl {

#define AURA_BACKEND_SHARED_BLOCK_HPP_GUARD
#include <aura/backend/shared/block.hpp>
#undef AURA_BACKEND_SHARED_BLOCK_HPP_GUARD

} // opencl 
} // backend_detail
} // aura


#endif // AURA_BACKEND_OPENCL_BLOCK_HPP

