#ifndef AURA_DETAIL_SIZE_HPP
#define AURA_DETAIL_SIZE_HPP

#include <vector>

namespace aura 
{

/// get the size of a contiguous block of memory
template <typename T, typename Allocator>
std::size_t size(std::vector<T, Allocator>& vec)
{
	return vec.size();
}

/// get the size of a contiguous block of memory
template <typename T>
std::size_t size(T* hptr)
{
	AURA_ERROR("a pointer does not define a range");
}


} // namespace aura

#endif // AURA_DETAIL_SIZE_HPP

