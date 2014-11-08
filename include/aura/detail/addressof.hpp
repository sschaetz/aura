#ifndef AURA_DETAIL_ADDRESSOF_HPP
#define AURA_DETAIL_ADDRESSOF_HPP

#include <vector>

namespace aura 
{

/// get the starting address of a contiguous block of memory
template <typename T, typename Allocator>
T* addressof(std::vector<T, Allocator>& vec)
{
	return &vec[0];
}

/// get the starting address of a contiguous block of memory
template <typename T>
T* addressof(T* hptr)
{
	return hptr;
}



} // namespace aura

#endif // AURA_DETAIL_ADDRESSOF_HPP

