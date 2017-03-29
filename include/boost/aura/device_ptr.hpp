#pragma once

#include <boost/functional/hash.hpp>

#if defined AURA_BASE_CUDA
#include <boost/aura/base/cuda/device_ptr.hpp>
#elif defined AURA_BASE_OPENCL
#include <boost/aura/base/opencl/device_ptr.hpp>
#elif defined AURA_BASE_METAL
#include <boost/aura/base/metal/device_ptr.hpp>
#endif

namespace boost
{
namespace aura
{

#if defined AURA_BASE_CUDA
namespace base = base_detail::cuda;
#elif defined AURA_BASE_OPENCL
namespace base = base_detail::opencl;
#elif defined AURA_BASE_METAL
namespace base = base_detail::metal;
#endif

using base::device_ptr;
using base::device_malloc;
using base::device_free;


} // namespace aura
} // namespace boost

namespace std
{

template<typename T>
struct hash<boost::aura::device_ptr<T>>
{
        typedef boost::aura::device_ptr<T> argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const
        {
                result_type hash = 0;

                const result_type h0 = s.get_base_ptr().hash();
                boost::hash_combine(hash, h0);

                result_type const h1 = s.get_offset();
                boost::hash_combine(hash, h1);

                result_type const h2 = s.get_device().get_ordinal();
                boost::hash_combine(hash, h2);

                return hash;
        }
};

} // namespace std
