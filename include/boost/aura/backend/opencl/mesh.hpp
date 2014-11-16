#ifndef AURA_BACKEND_OPENCL_MESH_HPP
#define AURA_BACKEND_OPENCL_MESH_HPP

#include <boost/aura/detail/svec.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace opencl {

typedef svec<std::size_t, AURA_MAX_MESH_DIMS> mesh;

} // opencl 
} // backend_detail
} // aura
} // boost


#endif // AURA_BACKEND_OPENCL_MESH_HPP

