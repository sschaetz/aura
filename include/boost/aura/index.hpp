#ifndef AURA_INDEX_HPP
#define AURA_INDEX_HPP

#include <boost/aura/detail/svec.hpp>

namespace boost
{
namespace aura {

typedef svec<int> index;

// let's try this and see if it blows up
constexpr int _ = -1;

} // namespace aura
} // boost

#endif // AURA_INDEX_HPP

