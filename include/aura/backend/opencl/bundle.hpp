#ifndef AURA_BACKEND_OPENCL_BUNDLE_HPP
#define AURA_BACKEND_OPENCL_BUNDLE_HPP

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

typedef svec<std::size_t, AURA_MAX_BUNDLE_DIMS> bundle;

} // opencl 
} // backend_detail
} // aura


#endif // AURA_BACKEND_OPENCL_BUNDLE_HPP

