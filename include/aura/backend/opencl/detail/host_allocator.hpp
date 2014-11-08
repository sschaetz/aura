#ifndef AURA_BACKEND_OPENCL_DETAIL_HOST_ALLOCATOR_HPP
#define AURA_BACKEND_OPENCL_DETAIL_HOST_ALLOCATOR_HPP


#include <aura/backend/opencl/mark.hpp>
#include <aura/backend/opencl/feed.hpp>

namespace aura
{
namespace backend_detail
{
namespace opencl 
{
namespace detail
{


template <typename T>
struct host_allocator
{
	/// create allocator
	typedef T value_type;
	host_allocator(feed& f, memory_tag mt = memory_tag::rw) : 
		mt_(mt), f_(&f)
	{}

	/// copy construct allocator
	template <typename T>
	host_allocator(const host_allocator<T>& other) : 
		f_(other.f_), mt_(other.mt_), map_(other.map_)
	{}

	/// allocate memory
	T* allocate(std::size_t n)
	{
		cl_mem_flags flag = translate_memory_tag(mt_);
		int errorcode = 0;
		memory m = clCreateBuffer(f.get_backend_context(), 
				flag | CL_MEM_ALLOC_HOST_PTR, 
				n, 0, &errorcode);
		AURA_OPENCL_CHECK_ERROR(errorcode);
		return map(m, n);
	}

	/// free memory
	void deallocate(T* p, std::size_t n)
	{
		memory m = unmap(p, n);
		AURA_OPENCL_SAFE_CALL(clReleaseMemObject(m));
		map_.erase((void*)p);
	}

	/// map OpenCL memory buffer into host memory space
	T* map(memory m, std::size_t n)
	{
		int errorcode = 0;
		T* r = (T*)clEnqueueMapBuffer(f_->get_backend_stream(), 
				m, CL_FALSE, 
				translate_memory_tag_inverted(mt_), 
				0,
				n*sizeof(T),
				0, NULL, NULL, &errorcode);
		AURA_OPENCL_CHECK_ERROR(errorcode);
		map_[(void*)r] = m;	
		return r;
	}

	/// unmap OpenCL memory buffer from  host memory space
	memory unmap(T* p)
	{
		memory m = map_[(void*)p];
		AURA_OPENCL_SAFE_CALL(clEnqueueUnmapMemObject(
					f_->get_backend_stream(),
					m, p, 0, NULL, NULL));
	}

private:
	/// feed through which mapping is done
	feed* f_;
	/// should memory be read/written by device
	memory_tag mt_;
	/// maps host pointer to device memory
	std::map<void*, memory> map_;

public:
	template <class T, class U>
	friend bool operator==(const host_allocator<T>& lhs, 
			const host_allocator<U>& rhs);
	template <class T, class U>
	friend bool operator!=(const host_allocator<T>& lhs, 
			const host_allocator<U>& rhs);
};

template <class T, class U>
bool operator==(const host_allocator<T>& lhs, const host_allocator<U>& rhs) 
{
	return *lhs.f_ == *rhs.f_ && 
		lhs.mt_ == rhs.mt_ && 
		lhs.map_ == rhs.map_;
}

template <class T, class U>
bool operator!=(const host_allocator<T>& lhs, const host_allocator<U>& rhs) 
{
	return !(lhs == rhs);
}

} // detail
} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_DETAIL_FEED_MARKER_HELPER_HPP

