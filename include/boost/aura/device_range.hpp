#ifndef AURA_DEVICE_RANGE_HPP
#define AURA_DEVICE_RANGE_HPP

#include <cstddef>
#include <boost/aura/bounds.hpp>
#include <boost/aura/slice.hpp>
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
    typedef T value_type;
	typedef device_ptr<T> iterator;
	typedef const device_ptr<T> const_iterator;
	
public:
	/// create empty range 
	device_range()  {}

	/// create one-dimensional range of size on device
	device_range(device_array<T>& da, slice idx)
	{
		std::size_t offset;
		std::tie(offset, bounds_) = 
			get_offset_and_bounds(da.get_bounds(), idx);
		ptr_ = backend::device_malloc_dependent<T>(da.begin(),
				(std::size_t)product(bounds_), offset);
	}

	/// create one-dimensional range of size on device
	device_range(device_array<T>& da, slice idx, bounds b)
	{
		std::size_t offset;
		std::tie(offset, bounds_) = 
			get_offset_and_bounds(da.get_bounds(), idx);
		bounds_ = b;
		ptr_ = backend::device_malloc_dependent<T>(da.begin(),
				product(bounds_), offset);
	}

    // TILWARNING: FIXME: THIS DOESNT AUTOMATICALLY CONSTRUCT A CONST DEVICE_RANGE :(
    device_range(const device_array<T>& da, slice idx, bounds b)
    {
        std::size_t offset;
        std::tie(offset, bounds_) =
            get_offset_and_bounds(da.get_bounds(), idx);
        bounds_ = b;
        ptr_ = backend::device_malloc_dependent<T>(da.begin(),
                product(bounds_), offset);
    }


    // TILWARNING: not really sure, if this is correct, but it works so far
    device_range<T>& operator=( const device_range<T> &dr)
    {
        bounds_ = dr.bounds_;
        ptr_ = backend::device_malloc_dependent<T>(dr.ptr_,
                        product(bounds_), 0);
        return *this;
    }


	/// destroy object
	~device_range() 
	{
		if (ptr_ != nullptr)
		{
			backend::device_free_dependent(ptr_);
		}
	}

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
	std::size_t size() const 
	{
		return product(bounds_);
	}

	/// return bounds
	const bounds & get_bounds() const 
	{
		return bounds_;
	}

	/// return copy of bounds
	bounds get_bounds() 
	{
		return bounds_;
	}

	/// return device
	const device& get_device() const
	{
		return ptr_.get_device();
	}

	device& get_device()
	{
		return ptr_.get_device();
	}

private:
	/// return offset and bounds from a slice
	std::tuple<std::size_t, bounds> get_offset_and_bounds(
			bounds b, slice idx)
	{
		bounds ret;
		std::size_t offset = 0;
		for (std::size_t i=0; i<idx.size(); i++) {
			if (idx[i] == _) {
				ret.push_back(b[i]);
			}
			if (i > 0) {
				idx[i-1] = b[i-1];
			}
			if (idx[i] != _) {
				offset += product(take(i+1, idx));
			} 
		}

		if (ret.size() == 0) {
			ret.push_back(1);
		}
		return std::make_tuple(offset, ret);
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

