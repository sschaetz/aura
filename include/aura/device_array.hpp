#ifndef AURA_DEVICE_ARRAY_HPP
#define AURA_DEVICE_ARRAY_HPP

#include <cstddef>
#include <boost/move/move.hpp>
#include <aura/device_buffer.hpp>

namespace aura {

/// continuous block of memory holdin multiple instances of a type T
template <typename T>
class device_array {

public:
	typedef device_ptr<T> iterator;
	typedef const device_ptr<T> const_iterator;
	
private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(device_array)

public:
	/// create empty array 
	device_array() : data_() {}

	/// create array of size on device
	device_array(std::size_t size, backend::device & d) :
		data_(size, d) 
	{}

	/// destroy object
	~device_array() 
	{}

	/**
	 * move constructor, move device_array here, invalidate other
	 *
	 * @param db device_array to move here
	 */
	device_array(BOOST_RV_REF(device_array) da) :
		data_(da.data_)
	{
	}

	/**
	 * move assignment, move device_array here, invalidate other
	 *
	 * @param d device to move here
	 */
	device_array & operator=(BOOST_RV_REF(device_array) da)
	{
		data_ = da.data_;
		return *this;
	}

private:
	/// data in the array
	device_buffer data_;
};

} // namespace arua

#endif // AURA_DEVICE_ARRAY_HPP

