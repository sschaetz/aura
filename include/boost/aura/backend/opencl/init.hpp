#ifndef AURA_BACKEND_OPENCL_INIT_HPP
#define AURA_BACKEND_OPENCL_INIT_HPP

#include <boost/aura/misc/deprecate.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace opencl {

inline void init() {}
inline void initialize() {}

DEPRECATED(void init());

} // opencl 
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_OPENCL_INIT_HPP

