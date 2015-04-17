#ifndef AURA_DEVICE_BUFFER_HPP
#define AURA_DEVICE_BUFFER_HPP

#include <cstddef>
#include <boost/move/move.hpp>
#include <boost/aura/backend.hpp>

namespace boost
{
namespace aura 
{

/// continuous block of memory holding multiple instances of a type T
template <typename T>
class device_buffer 
{

public:
	typedef device_ptr<T> iterator;
	typedef const device_ptr<T> const_iterator;
	
private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(device_buffer)

public:
	/// create empty buffer
	device_buffer() : ptr_(), size_(0) 
	{}

	/// create buffer of size on device
	device_buffer(std::size_t size, backend::device & d) :
		ptr_(backend::device_malloc<T>(size, d)), size_(size) 
	{}

	/// destroy object
	~device_buffer() 
	{
		finalize();
	}

	/**
	 * move constructor, move device here, invalidate other
	 *
	 * @param db device_buffer to move here
	 */
	device_buffer(BOOST_RV_REF(device_buffer) db) :
        ptr_(std::move(db.ptr_)), size_(std::move(db.size_))
	{
		db.ptr_.invalidate();
		db.size_ = 0;
	}

	/**
	 * move assignment, move device_buffer here, invalidate other
	 *
	 * @param d device to move here
	 */
	device_buffer& operator=(BOOST_RV_REF(device_buffer) db)
	{
        finalize();
        ptr_ = std::move(db.ptr_);
		size_ = db.size_;
		db.ptr_.invalidate();
		db.size_ = 0;
		return *this;
	}

	// resize the buffer to contain size elements on device d
	void resize(const std::size_t size, device& d)
	{
		finalize();
		ptr_ = backend::device_malloc<T>(size, d);
		size_ = size;
	}
	
	// resize the buffer to contain size elements
	void resize(const std::size_t size)
	{
		// we need to make sure we know which device so we need
		// an existing pointer for this signature
		assert(ptr_ != nullptr);
		resize(size, ptr_.get_device());
	}

	/// return beginning of buffer
	iterator begin()
	{
		return ptr_;
	}
	const iterator begin() const
	{
		return ptr_;
	}
	
	/// return end of buffer
	iterator end()
	{
		return ptr_+size_;
	}
	const iterator end() const
	{
		return ptr_+size_;
	}

	/// return pointer to underlying array
	T* data()
	{
		return (T*)ptr_.get();
	}

	/// return pointer to underlying array
	const T* data() const
	{
		return (T*)ptr_.get();
	}

	/// return beginning of buffer as raw pointer
	T * begin_ptr()
	{
		return (T*)ptr_.get();
	}
	const T * begin_ptr() const
	{
		return (T*)ptr_.get();
	}
	
	/// return end of buffer as raw pointer 
	T * end_ptr()
	{
		return (T*)ptr_.get() + size_;
	}
	const T * end_ptr() const
	{
		return (T*)ptr_.get() + size_;
	}

	/// return size of buffer
	std::size_t size() const 
	{
		return size_;
	}

	/// return device
	const device& get_device () const
	{
		return ptr_.get_device();
	}
	device& get_device()
	{
		return ptr_.get_device();
	}

private:
	/// finalize object (called from dtor and move assign)
	void finalize() 
	{
		if(nullptr != ptr_) {
			backend::device_free<T>(ptr_);
		}
		ptr_.invalidate();
		size_ = 0;
	}

private:
	/// pointer to device memory
	backend::device_ptr<T> ptr_;
	/// size of buffer
	std::size_t size_;
};

} // namespace aura
} // boost

#endif // AURA_DEVICE_BUFFER_HPP

