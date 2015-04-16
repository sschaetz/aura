#ifndef AURA_META_TSIZEOF_HPP
#define AURA_META_TSIZEOF_HPP

namespace boost {
namespace aura {

/// Compiletime size of pack of types.
template<typename... Targs>
struct tsizeof;

template<typename T0>
struct tsizeof<T0>
{
	enum ts : size_t {sz = sizeof(T0)};
};

template<typename T0, typename... Targs>
struct tsizeof<T0,Targs...>
{
	enum ts : size_t { sz = sizeof(T0) + tsizeof<Targs...>::sz };
};

} // namespace aura
} // namespace boost

#endif // AURA_META_TSIZEOF_HPP

