#pragma once

namespace boost
{
namespace aura
{

/// Mesh and bundle suggestions that can be used when invoking kernels.
using mesh = std::array<std::size_t, 3>;
using bundle = std::array<std::size_t, 3>;

} // namespace aura
} // boost
