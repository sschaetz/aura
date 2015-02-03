#ifndef AURA_SLICE_HPP
#define AURA_SLICE_HPP

#include <boost/aura/detail/svec.hpp>

namespace boost
{
namespace aura {

typedef svec<int> slice;

// let's try this and see if it blows up
constexpr int _ = -1;

} // namespace aura
} // boost

#endif // AURA_SLICE_HPP

