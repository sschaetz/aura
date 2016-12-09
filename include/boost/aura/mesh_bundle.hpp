#pragma once

namespace boost
{
namespace aura
{

/// Mesh and bundle suggestions that can be used when invoking kernels.
using mesh = std::array<std::size_t, 3>;
using bundle = std::array<std::size_t, 3>;

/// Eagerly create a mesh from a type:
///   type must reports its size with size() method,
///   type must have begin() end() iterator methods
///   type istance must be of size 3 or less
template <typename T>
mesh make_mesh(const T& v)
{
        assert(v.size() <= 3);
        mesh m;
        std::copy(v.begin(), v.end(), m.begin());
        return m;
}

/// Eagerly create a bundle from a type:
///   type must reports its size with size() method,
///   type must have begin() end() iterator methods
///   type istance must be of size 3 or less
template <typename T>
bundle make_bundle(const T& v)
{
        assert(v.size() <= 3);
        bundle b;
        std::copy(v.begin(), v.end(), b.begin());
        return b;
}

} // namespace aura
} // boost
