#ifndef AURA_BACKEND_OPENCL_DETAIL_HOST_ALLOCATOR_HPP
#define AURA_BACKEND_OPENCL_DETAIL_HOST_ALLOCATOR_HPP

#include <boost/bimap.hpp>
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
template< class MapType >
void print_map(const MapType & m)
{
    typedef typename MapType::const_iterator const_iterator;
    for( const_iterator iter = m.begin(), iend = m.end(); iter != iend; ++iter )
    {
        std::cout << iter->first << "-->" << iter->second << std::endl;
    }
}

template <typename T>
struct host_allocator
{
private:
	typedef boost::bimaps::bimap<
		boost::bimaps::set_of<memory>,
		boost::bimaps::set_of<void*>,
		boost::bimaps::with_info<std::size_t>
	> bm_t;

	typedef bm_t::value_type mapping;

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
		std::cout << "allocated " << m << std::endl;
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
		std::cout << "unmapping " << m << " " << p <<
			" (size " << map_.left.find(m)->info << 
			")" << std::endl;
		AURA_OPENCL_SAFE_CALL(clEnqueueUnmapMemObject(
					f_->get_backend_stream(),
					m, 
					p, 0, NULL, NULL));
		return m;
	}

	void debug_map()
	{
		print_map(map_.left);
	}

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
		std::cout << "mapping " << m << " (size " << n*sizeof(T) << ")" << 
			" to " << p << std::endl;
		map_.insert(mapping(m, p, n));
		std::cout << "map contents: " << map_.left.at(m) << std::endl;
		return p;
	}

private:
	/// feed through which mapping is done
	feed* f_;
	/// should memory be read/written by device
	memory_tag mt_;
	/// maps host pointer to device memory
	bm_t map_;

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



/// I'm scared:
namespace std
{

template <typename U> 
struct allocator_traits<
		::aura::backend_detail::opencl::detail::host_allocator<U>
	>
{
	typedef ::aura::backend_detail::opencl::detail::host_allocator<U> 
		allocator_type;

	typedef U value_type;

	typedef U* pointer;
	typedef U* const const_pointer;
	typedef void* void_pointer;
	typedef void* const const_void_pointer;

	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	typedef true_type propagate_on_container_copy_assignment;
	typedef true_type propagate_on_container_move_assignment;
	typedef true_type propagate_on_container_swap;

	template <typename T> using rebind_alloc = allocator_type;
	template <typename T> using rebind_traits = 
		allocator_traits<rebind_alloc<T>>;

	static pointer allocate(allocator_type& a, size_type n)
	{
		return a.allocate(n);
	}
	static pointer allocate(allocator_type& a, size_type n, 
			const_void_pointer hint)
	{
		return a.allocate(n);
	}

	static void deallocate(allocator_type& a, pointer p, size_type n)
	{
		return a.deallocate(p, n);
	}

	template <typename T, typename... Args>
	static void construct(allocator_type& a, T* p, Args&&... args)
	{
		::new (static_cast<void*>(p)) T(forward<Args>(args)...);
	}

	template <typename T>
	static void destroy(allocator_type& a, T* p) 
	{
		p->~T();
	}

	static size_type max_size(const allocator_type& a)
	{
		return numeric_limits<size_type>::max();
	}

	static allocator_type select_on_container_copy_construction(
			const allocator_type& rhs)
	{
		return rhs;
	}
};

} // namespace std


#endif // AURA_BACKEND_OPENCL_DETAIL_FEED_MARKER_HELPER_HPP

