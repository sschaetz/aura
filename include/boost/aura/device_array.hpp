#ifndef AURA_DEVICE_ARRAY_HPP
#define AURA_DEVICE_ARRAY_HPP

#include <cstddef>
#include <boost/move/move.hpp>
#include <boost/aura/device_buffer.hpp>
#include <boost/aura/device_range.hpp>
#include <boost/aura/bounds.hpp>
#include <boost/aura/copy.hpp>

namespace boost
{
namespace aura 
{

template <typename T>
class device_range;

/// continuous block of memory holding multiple instances of a type T
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

	/// create one-dimensional array of size on device
	device_array(std::size_t size, backend::device & d) :
		bounds_(size), data_(size, d) 
	{}
	
	/// create multi-dimensional array of size on device
	device_array(const bounds & b, backend::device & d) :
		bounds_(b), data_(product(b), d) 
	{}

	/// create one-dimensional array from std vector on device
	device_array(const std::vector<T>& vec, 
			backend::device & d, backend::feed & f) :
		bounds_(vec.size()), data_(vec.size(), d) 
	{
		copy(vec, *this, f);
	}

	/// create multi-dimensional array from std vector on device
	device_array(const std::vector<T>& vec, const bounds & b,
			backend::device & d, backend::feed & f) :
		bounds_(b), data_(vec.size(), d) 
	{
		assert(product(b) == vec.size());
		copy(vec, *this, f);
	}

	/// destroy object
	~device_array() 
	{}

	/**
	 * move constructor, move device_array here, invalidate other
	 *
	 * @param db device_array to move here
	 */
	device_array(BOOST_RV_REF(device_array) da) :
		bounds_(da.bounds_), data_(da.data_)
	{
		da.bounds_.clear();
	}

	/**
	 * move assignment, move device_array here, invalidate other
	 *
	 * @param d device to move here
	 */
	device_array& operator=(BOOST_RV_REF(device_array) da)
	{
		bounds_ = da.bounds_;
		data_ = boost::move(da.data_);
		da.bounds_.clear();
		return *this;
	}

	device_range<T> operator [](slice i)
	{
		return device_range<T>(*this, i);
	}

	// resize the buffer to contain size elements on device d
	void resize(const std::size_t size, device& d)
	{
		data_.resize(size, d);
		bounds_ = bounds(size);
	}
	
	// resize the buffer to contain size elements
	void resize(const std::size_t size)
	{
		data_.resize(size);
		bounds_ = bounds(size);
	}

	// resize the buffer to contain bounds elements on device d
	void resize(const bounds b, device& d)
	{
		data_.resize(product(b), d);
		bounds_ = b;
	}
	
	// resize the buffer to contain bounds elements
	void resize(const bounds b)
	{
		data_.resize(product(b));
		bounds_ = b;
	}

	/// return beginning of buffer
	iterator begin() const
	{
		return data_.begin();
	}
	
	/// return end of buffer
	iterator end() const
	{
		return data_.end();
	}

	/// return pointer to underlying array
	T* data()
	{
		return data_.data();
	}

	/// return pointer to underlying array
	const T* data() const
	{
		return data_.data();
	}

	/// return beginning of array as raw pointer
	T * begin_ptr() const
	{
		return data_.begin_ptr();
	}
	
	/// return end of array as raw pointer 
	T * end_ptr() const
	{
		return data_.end_ptr();
	}
	
	/// return number of elements in array
	std::size_t size() const {
		return data_.size();
	}

	/// return bounds
	const bounds & get_bounds() const {
		return bounds_;
	}

	/// return copy of bounds
	bounds get_bounds() {
		return bounds_;
	}

	/// return device
	device& get_device()
	{
		return data_.get_device();
	}

private:
	/// bounds of array
	bounds bounds_;

	/// data in the array
	device_buffer<T> data_;
};

} // namespace aura
} // boost

#endif // AURA_DEVICE_ARRAY_HPP

