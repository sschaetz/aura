#pragma once

#include <boost/aura/memory_tag.hpp>
#include <boost/aura/device.hpp>

#include <array>

namespace boost
{
namespace aura
{

/// Mesh and bundle suggestions that can be used when invoking kernels.
using mesh = std::array<std::size_t, 3>;
using bundle = std::array<std::size_t, 3>;

} // aura
} // boost
