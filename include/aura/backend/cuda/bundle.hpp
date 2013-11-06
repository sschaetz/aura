#ifndef AURA_BACKEND_CUDA_BUNDLE_HPP
#define AURA_BACKEND_CUDA_BUNDLE_HPP

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

typedef svec<std::size_t, AURA_MAX_BUNDLE_DIMS> block;

} // cuda 
} // backend_detail
} // aura


#endif // AURA_BACKEND_CUDA_BUNDLE_HPP

