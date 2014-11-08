#ifndef AURA_DEVICE_VIEW_HPP
#define AURA_DEVICE_VIEW_HPP

#include <tuple>
#include <cstddef>
#include <boost/move/move.hpp>
#include <boost/any.hpp>
#include <aura/backend.hpp>
#include <aura/error.hpp>
#include <aura/detail/addressof.hpp>
#include <aura/detail/size.hpp>

namespace aura {


/// forward declaration
template <typename T>
struct device_view;

/// create a device_view by mapping a C to a device
template <typename T, typename C>
device_view<T> map(C& c, device& d)
{
	device_view<T> r;
	r.ptr_ = device_map<T>(addressof(c), size(c), d);
	r.hptr_ = addressof(c);
	r.size_ = size(c);
	r.moved_from_obj_ = std::move(c);
	return r;
}

/// unmap a previously mapped C
template <typename T, typename C>
void unmap(device_view<T>& v, C& c, feed& f)
{
	device_unmap(v.hptr_, v.ptr_, v.size(), f);
	c = boost::any_cast<C>(std::move(v.moved_from_obj_));
	v.hptr_ = nullptr;
	v.ptr_ = nullptr;
	v.size_ = 0;
}



template <typename T>
class device_view 
{

public:
	typedef device_ptr<T> iterator;
	typedef const device_ptr<T> const_iterator;

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(device_view)

public:
	/// create an empty device_view
	device_view() : ptr_(nullptr), hptr_(nullptr), size_(0) {}

	/// construct a device_view on a C
	template <typename C> 
	device_view(C& c, device& d) : 
		ptr_(device_map<T>(addressof(c), size(c), d)),
		hptr_(addressof(c)),
		size_(size(c)),
		moved_from_obj_(std::move(c))
	{}

	/// construct a device_view on a range of Cs
	template <typename C> 
	device_view(C& start, C& end, device& d) : 
		ptr_(device_map<T>(addressof(start), 
					std::distance(start,end), d)),
		hptr_(addressof(start)),
		size_(std::distance(start,end)),
		moved_from_obj_(std::move(std::tuple<C,C>(start,end)))
	{}

	/// move construct a device_view
	device_view(BOOST_RV_REF(device_view) v) : 
		ptr_(v.ptr_),
		hptr_(v.hptr_),
		size_(v.size_),
		moved_from_obj_(std::move(v.moved_from_obj_))
	{
		v.ptr_ = nullptr;
		v.hptr_ = nullptr;
		v.size_ = 0;
	}
	
	/// move assign a device_view
	device_view<T>& operator= (BOOST_RV_REF(device_view<T>) other)
	{
		ptr_ = other.ptr_;
		hptr_ = other.hptr_;
		size_ = other.size_;
		moved_from_obj_ = std::move(other.moved_from_obj_);
		other.ptr_ = nullptr;
		other.hptr_ = nullptr;
		other.size_ = 0;
	}
	
	/// destroy (unmap) a device_view
	~device_view() 
	{
	}

	/// return beginning of device_view 
	iterator begin() const
	{
		return ptr_;
	}
	
	/// return end of device_view 
	iterator end() const
	{
		return ptr_+size_;
	}

	/// return beginning of device_view as raw pointer
	T* begin_ptr() const
	{
		return (T*)ptr_.get();
	}
	
	/// return end of device_view as raw pointer 
	T* end_ptr() const
	{
		return ((T*)ptr_.get()) + size_;
	}

	/// return size of device_view 
	std::size_t size() const 
	{
		return size_;
	}

	void debug() 
	{
		std::cout << "ptr_: " << ptr_.get() << 
			" hptr_: " << hptr_ << 
			" size_: " << size_ << 
			std::endl;
	}

	device_ptr<T> ptr_;
	T* hptr_;
	std::size_t size_;
	boost::any moved_from_obj_;
};

} // namespace arua

#endif // AURA_DEVICE_VIEW_HPP

