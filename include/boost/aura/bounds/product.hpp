#pragma once

#include <numeric>
#include <type_traits>

namespace boost
{
namespace aura
{

/// Identify element for product.
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
product_identity_element()
{
        return 1;
}

/// Identify element for product.
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
addition_identity_element()
{
        return 0;
}

/// Compute the product of the values in a container.
template <typename ContainerT>
auto product(const ContainerT& c) -> typename ContainerT::value_type
{
        using T = typename ContainerT::value_type;

        if (c.begin() == c.end())
        {
                return addition_identity_element<T>();
        }

        return std::accumulate(c.begin(), c.end(),
                product_identity_element<T>(), std::multiplies<T>());
}

} // namespace aura
} // namespace boost
