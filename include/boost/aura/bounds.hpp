#pragma once

#include <boost/aura/bounds/product.hpp>
#include <boost/aura/bounds/tiny_vector.hpp>
#include <boost/aura/config.hpp>

namespace boost
{
namespace aura
{

using bounds = tiny_vector<std::size_t, AURA_TINY_VECTOR_MAX_SIZE>;

} // namespace aura
} // namespace boost
