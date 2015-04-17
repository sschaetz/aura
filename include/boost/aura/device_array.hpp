#ifndef AURA_DEVICE_ARRAY_HPP
#define AURA_DEVICE_ARRAY_HPP

#include <numeric>
#include <cstddef>

#include <boost/move/move.hpp>
#include <boost/aura/device_buffer.hpp>
#include <boost/aura/device_range.hpp>
#include <boost/aura/bounds.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/backend.hpp>


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

	/// create multi-dimensional array from an iterator and std::vector bounds on device
	template <typename Iterator>
	device_array(Iterator it, const std::vector<std::size_t>& b,
			backend::device & d, backend::feed & f)
		: bounds_(b)
		, data_(std::accumulate(b.begin(), b.end(),
			1, std::multiplies<std::size_t>()), d)
	{
		copy(it, *this, f);
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
        bounds_(std::move(da.bounds_)), data_(std::move(da.data_))
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

	// create a device range from a slice (calculate bounds automatically)
	device_range<T> operator()(slice s)
	{
		return device_range<T>(*this, s);
	}

	// create a device range from a slice and bounds
	device_range<T> operator()(slice s, bounds b)
	{
		return device_range<T>(*this, s, b);
	}

	// create a device range from a slice and bounds
	const device_range<T> operator()(slice s, bounds b) const
	{
		return device_range<T>(*this, s, b);
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

    // reshape the bounds
    void reshape(const bounds b)
    {
        assert(product(b) == product(bounds_));
        bounds_ = b;
    }

    // squeezes the bounds
    void squeeze(void)
    {
        size_t j = 0;
        for (size_t i = 0; i < bounds_.size(); i++)
        {
            if (bounds_[i] > 1)
            {
                bounds_[j] = bounds_[i];
                if (i > j)
                    bounds_[i] = 1;
                j++;
            }
        }

    }

	// get a single value in the vector (synchronous!)
	T get_value (std::size_t index, backend::feed& f) const
	{
		T value;
		boost::aura::backend::copy<T>(&value,data_.begin()+index,1, f);
		wait_for(f);
		return value;
	}

    // get the value in a vector representing a scalar (synchronous!)
    T get_value (backend::feed& f) const
    {
        assert(1 == data_.size());
        T value;
        boost::aura::backend::copy<T>(&value,data_.begin(),1, f);
        wait_for(f);
        return value;
    }

	// set a single value in the vector (synchronous!)
	void set_value(std::size_t index, T value, backend::feed& f)
	{
		boost::aura::backend::copy<T>(data_.begin()+index,&value,1, f);
		wait_for(f);
	}

    // set the value in a vector representing a scalar (synchronous!)
    void set_value(T value, backend::feed& f)
    {
        assert(1 == data_.size());
        boost::aura::backend::copy<T>(data_.begin(),&value,1, f);
        wait_for(f);
    }

    // copy the full vector to the host and return it (synchronous!)
    std::vector<T> get_host_vector (backend::feed& f) const
    {
        std::vector<T> host_vector(data_.size());
        copy(*this, host_vector, f);
        wait_for(f);
        return host_vector;
    }


	/// return beginning of buffer
	iterator begin()
	{
		return data_.begin();
	}
	const iterator begin() const
	{
		return data_.begin();
	}

	/// return end of buffer
	iterator end()
	{
		return data_.end();
	}
	const iterator end() const
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
	T * begin_ptr()
	{
		return data_.begin_ptr();
	}
	const T * begin_ptr() const
	{
		return data_.begin_ptr();
	}

	/// return end of array as raw pointer
	T * end_ptr()
	{
		return data_.end_ptr();
	}
	const T * end_ptr() const
	{
		return data_.end_ptr();
	}

	/// return number of elements in array
	std::size_t size() const
	{
		return data_.size();
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
		return data_.get_device();
	}

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

