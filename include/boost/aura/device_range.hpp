#ifndef AURA_DEVICE_RANGE_HPP
#define AURA_DEVICE_RANGE_HPP

#include <cstddef>
#include <boost/aura/bounds.hpp>
#include <boost/aura/index.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>


namespace boost
{
namespace aura 
{

template <typename T>
class device_array;

/// a range of device memory
template <typename T>
class device_range 
{

public:
	typedef device_ptr<T> iterator;
	typedef const device_ptr<T> const_iterator;
	
public:
	/// create empty range 
	device_range()  {}

	/// create one-dimensional range of size on device
	device_range(device_array& da, index idx) :
	{
		const bounds b = da.get_bounds();
		bool locked = false;	
		// initialize offset and bounds
		std::size_t offset = b[0];
		if (idx[0] != _) {
			locked = true;
			offset = idx[0];
		}
		// calculate the increment and the bounds
		for (int i=1; i<b.size(); i++) {
			if (idx[i] != _) {
				locked = true;
				offset += idx[i];
			}
		}
		// take the pointer of the array, increment it, store it
		// calculate the bounds
	}

	/// destroy object
	~device_range() 
	{}

	/// return beginning of buffer
	iterator begin() const
	{
		return ptr_;
	}
	
	/// return end of buffer
	iterator end() const
	{
		return ptr_+product(bounds_);
	}

	/// return pointer to underlying data 
	T* data()
	{
		return ptr_.get();
	}

	/// return pointer to underlying data 
	const T* data() const
	{
		return ptr_.get();
	}
	
	/// return number of elements in range 
	std::size_t size() const {
		return product(bounds_);
	}

	/// return bounds
	const bounds & get_bounds() const {
		return bounds_;
	}

	/// return copy of bounds
	bounds get_bounds() {
		return bounds_;
	}

private:
	/// pointer to device memory
	backend::device_ptr<T> ptr_;
	
	/// bounds of array
	bounds bounds_;
};

} // namespace aura
} // boost

#endif // AURA_DEVICE_RANGE_HPP

