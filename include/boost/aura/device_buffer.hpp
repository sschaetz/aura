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
		ptr_(db.ptr_), size_(db.size_) 
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
		ptr_ = db.ptr_;
		size_ = db.size_;
		db.ptr_.invalidate();
		db.size_ = 0;
		return *this;
	}
	
	/// return beginning of buffer
	iterator begin() const
	{
		return ptr_;
	}
	
	/// return end of buffer
	iterator end() const
	{
		return ptr_+size_;
	}

	/// return beginning of buffer as raw pointer
	T * begin_ptr() const
	{
		return (T*)ptr_.get();
	}
	
	/// return end of buffer as raw pointer 
	T * end_ptr() const
	{
		return (T*)ptr_.get() + size_;
	}

	/// return size of buffer
	std::size_t size() const 
	{
		return size_;
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

