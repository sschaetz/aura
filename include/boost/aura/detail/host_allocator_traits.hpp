#ifndef AURA_DETAIL_HOST_ALLOCATOR_TRAITS_HPP
#define AURA_DETAIL_HOST_ALLOCATOR_TRAITS_HPP

/// I'm scared:
namespace std
{

template <typename U> 
struct allocator_traits<::boost::aura::backend::detail::host_allocator<U>>
{
	typedef ::boost::aura::backend::detail::host_allocator<U> allocator_type;

	typedef U value_type;

	typedef U* pointer;
	typedef U* const const_pointer;
	typedef void* void_pointer;
	typedef void* const const_void_pointer;

	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	typedef false_type propagate_on_container_copy_assignment;
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

#endif // AURA_DETAIL_HOST_ALLOCATOR_TRAITS_HPP

