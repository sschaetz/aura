#ifndef AURA_BACKEND_CUDA_DETAIL_HOST_ALLOCATOR_HPP
#define AURA_BACKEND_CUDA_DETAIL_HOST_ALLOCATOR_HPP

#include <aura/backend/cuda/feed.hpp>

namespace aura
{
namespace backend_detail
{
namespace cuda 
{
namespace detail
{


template <typename T>
struct host_allocator
{
private:
	typedef std::map<memory, void*> map_t;

	typedef map_t::value_type mapping;

public:
	/// create allocator
	typedef T value_type;
	host_allocator(feed& f, memory_tag mt = memory_tag::rw) : 
		f_(&f), mt_(mt), map_(other.map_)
	{}

	/// copy construct allocator
	template <typename U>
	host_allocator(const host_allocator<U>& other) : 
		mt_(other.mt_), f_(other.f_)
	{}

	/// allocate memory
	T* allocate(std::size_t n)
	{
		void* ptr;
		f->set();  
		AURA_CUDA_SAFE_CALL(cuMemAllocHost(&ptr, n*sizeof(T)));
		f->unset();
	}

	/// free memory
	void deallocate(T* p, std::size_t n)
	{
		f->set();  
		AURA_CUDA_SAFE_CALL(cuMemFreeHost(ptr));
		f->unset();
	}

	/// map CUDA memory buffer into host memory space
	T* map(memory m)
	{
		return map_[m];
	}


	/// unmap CUDA memory buffer from host memory space
	memory unmap(T* p)
	{
		memory m;
		f->set();  
		AURA_CUDA_SAFE_CALL(cuMemHostGetDevicePointer(&m, (void*)p, 0));
		map_[m] = p;
		f->unset();
		return m;
	}

private:
	/// feed through which mapping is done
	feed* f_;
	/// should memory be read/written by device
	memory_tag mt_;
	/// maps device pointer to host memory
	bimap_t map_;

public:
	template <typename U, typename V>
	friend bool operator==(const host_allocator<U>& lhs, 
			const host_allocator<V>& rhs);
	template <typename U, typename V>
	friend bool operator!=(const host_allocator<U>& lhs, 
			const host_allocator<V>& rhs);
};

template <typename U, typename V>
bool operator==(const host_allocator<U>& lhs, const host_allocator<V>& rhs)
{
	return *lhs.f_ == *rhs.f_ && 
		lhs.mt_ == rhs.mt_;
}

template <typename U, typename V>
bool operator!=(const host_allocator<U>& lhs, const host_allocator<V>& rhs)
{
	return !(lhs == rhs);
}

} // detail
} // cuda 
} // backend_detail
} // aura


} // namespace std

#endif // AURA_BACKEND_CUDA_DETAIL_FEED_MARKER_HELPER_HPP

