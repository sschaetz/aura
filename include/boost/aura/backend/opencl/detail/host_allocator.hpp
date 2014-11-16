#ifndef AURA_BACKEND_OPENCL_DETAIL_HOST_ALLOCATOR_HPP
#define AURA_BACKEND_OPENCL_DETAIL_HOST_ALLOCATOR_HPP

#include <boost/bimap.hpp>
#include <boost/aura/backend/opencl/feed.hpp>


namespace boost
{
namespace aura
{
namespace backend_detail
{
namespace opencl 
{
namespace detail
{

#if 0
template< class MapType >
void print_map(const MapType & m)
{
	typedef typename MapType::const_iterator const_iterator;
	for( const_iterator iter = m.begin(), iend = m.end(); 
			iter != iend; ++iter)
	{
		std::cout << iter->first << "-->" << iter->second << std::endl;
	}
}
#endif

template <typename T>
struct host_allocator
{
private:
	typedef boost::bimaps::bimap<
		boost::bimaps::set_of<memory>,
		boost::bimaps::set_of<void*>,
		boost::bimaps::with_info<std::size_t>
	> bimap_t;

	typedef bimap_t::value_type mapping;

public:
	/// create allocator
	typedef T value_type;
	host_allocator(feed& f, memory_tag mt = memory_tag::rw) : 
		f_(&f), mt_(mt)
	{}

	/// copy construct allocator
	template <typename U>
	host_allocator(const host_allocator<U>& other) : 
		mt_(other.mt_), f_(other.f_), map_(other.map_)
	{}

	/// allocate memory
	T* allocate(std::size_t n)
	{
		cl_mem_flags flag = translate_memory_tag(mt_);
		int errorcode = 0;
		memory m = clCreateBuffer(f_->get_backend_context(), 
				flag | CL_MEM_ALLOC_HOST_PTR, 
				n*sizeof(T), 0, &errorcode);
		AURA_OPENCL_CHECK_ERROR(errorcode);
		return map_impl(m, n);
	}

	/// free memory
	void deallocate(T* p, std::size_t n)
	{
		memory m = unmap(p);
		AURA_OPENCL_SAFE_CALL(clReleaseMemObject(m));
		map_.erase(mapping(m, p));
	}

	/// map OpenCL memory buffer into host memory space
	T* map(memory m)
	{
		return map_impl(m, map_.left.find(m)->info);
	}


	/// unmap OpenCL memory buffer from  host memory space
	memory unmap(T* p)
	{
		memory m = map_.right.at(p);
		AURA_OPENCL_SAFE_CALL(clEnqueueUnmapMemObject(
					f_->get_backend_stream(),
					m, 
					p, 0, NULL, NULL));
		return m;
	}
#if 0
	void debug_map()
	{
		print_map(map_.left);
	}
#endif

private:
	/// implementation of map
	T* map_impl(memory m, std::size_t n)
	{
		int errorcode = 0;
		T* p = (T*)clEnqueueMapBuffer(f_->get_backend_stream(), 
				m, CL_FALSE,
				translate_map_tag_inverted(mt_), 
				0,
				n*sizeof(T),
				0, NULL, NULL, &errorcode);
		AURA_OPENCL_CHECK_ERROR(errorcode);
		map_.insert(mapping(m, p, n));
		return p;
	}

private:
	/// feed through which mapping is done
	feed* f_;
	/// should memory be read/written by device
	memory_tag mt_;
	/// maps host pointer to device memory and vice versa
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
		lhs.mt_ == rhs.mt_ && 
		lhs.map_ == rhs.map_;
}

template <typename U, typename V>
bool operator!=(const host_allocator<U>& lhs, const host_allocator<V>& rhs) 
{
	return !(lhs == rhs);
}

} // detail
} // opencl 
} // backend_detail
} // aura
} // boost



#endif // AURA_BACKEND_OPENCL_DETAIL_FEED_MARKER_HELPER_HPP

