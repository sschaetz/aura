#pragma once

#include <boost/aura/bounds/product.hpp>
#include <boost/aura/bounds/tiny_vector.hpp>
#include <boost/aura/config.hpp>

namespace boost
{
namespace aura
{

using bounds = tiny_vector<std::size_t, AURA_TINY_VECTOR_MAX_SIZE>;

/// Eagerly create bounds from a type:
///   type must reports its size with size() method,
///   type must have begin() end() iterator methods
///   type istance must be of size 3 or less
template <typename T>
bounds make_bounds(const T& v)
{
        assert(v.size() <= AURA_TINY_VECTOR_MAX_SIZE);
        bounds b;
        std::copy(v.begin(), v.end(), b.begin());
        return b;
}

} // namespace aura
} // namespace boost
