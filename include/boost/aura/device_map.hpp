#ifndef AURA_DEVICE_MAP_HPP
#define AURA_DEVICE_MAP_HPP

#include <tuple>
#include <cstddef>
#include <boost/move/move.hpp>
#include <boost/any.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/error.hpp>
#include <boost/aura/detail/addressof.hpp>
#include <boost/aura/detail/size.hpp>

namespace boost
{
namespace aura
{


/// forward declaration
template <typename T>
struct device_map;

#if 0
/// create a device_map by mapping a C to a device
template <typename T, typename C>
device_map<T> map(C& c, memory_tag tag, device& d)
{
	device_map<T> r;
	r.ptr_ = ::boost::aura::backend::device_map(addressof(c), size(c), tag, d);
	r.hptr_ = addressof(c);
	r.size_ = size(c);
	r.moved_from_obj_ = std::move(c);
	r.tag_ = tag;
	return r;
}

/// unmap a previously mapped C
template <typename T, typename C>
void unmap(device_map<T>& v, C& c, feed& f)
{
	device_unmap(v.hptr_, v.ptr_, v.size(), v.tag_, f);
	c = boost::any_cast<C>(std::move(v.moved_from_obj_));
	v.hptr_ = nullptr;
	v.ptr_ = nullptr;
	v.size_ = 0;
}

#endif

template <typename T>
class device_map
{

public:
	typedef device_ptr<T> iterator;
	typedef const device_ptr<T> const_iterator;

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(device_map)

public:
	/// create an empty device_map
	device_map() : dptr_(nullptr), hptr_(nullptr),
	size_(0), tag_(memory_tag::rw) {}

	/// construct a device_map on a C
	template <typename C>
	device_map(C& c, memory_tag tag, device& d) :
		dptr_(::boost::aura::backend::device_map_alloc(addressof(c),
					::boost::aura::size(c), tag, d)),
		hptr_(addressof(c)),
		size_(::boost::aura::size(c)),
		moved_from_obj_(std::move(c)),
		tag_(tag)
	{}

	/// construct a device_map on a range of Cs
	template <typename C>
	device_map(C& start, C& end, memory_tag tag, device& d) :
		dptr_(::boost::aura::backend::device_map_alloc(addressof(start),
					std::distance(start,end), tag, d)),
		hptr_(addressof(start)),
		size_(std::distance(start,end)),
		moved_from_obj_(std::move(std::tuple<C,C>(start,end))),
		tag_(tag)
	{}

	/// move construct a device_map
	device_map(BOOST_RV_REF(device_map) v) :
		dptr_(v.dptr_),
		hptr_(v.hptr_),
		size_(v.size_),
		moved_from_obj_(std::move(v.moved_from_obj_)),
		tag_(v.tag_)
	{
		v.dptr_ = nullptr;
		v.hptr_ = nullptr;
		v.size_ = 0;
	}

	/// move assign a device_map
	device_map<T>& operator= (BOOST_RV_REF(device_map<T>) other)
	{
		dptr_ = other.dptr_;
		hptr_ = other.hptr_;
		size_ = other.size_;
		moved_from_obj_ = std::move(other.moved_from_obj_);
		tag_ = other.tag_;
		other.dptr_ = nullptr;
		other.hptr_ = nullptr;
		other.size_ = 0;
	}

	/// destroy (unmap) a device_map
	~device_map()
	{
		::boost::aura::backend::device_map_free(hptr_, dptr_);
	}

	/// unmap data previously mapped to device memory
	template <typename C>
	void unmap(C& c, feed& f)
	{
		device_unmap(hptr_, dptr_, size(), f);
		c = boost::any_cast<C>(std::move(moved_from_obj_));
	}

	/// remap previously unmapped data back to device memory
	template <typename C>
	void remap(C& c, feed& f)
	{
		assert(::boost::aura::addressof(c) == hptr_);
		assert(::boost::aura::size(c) == size_);
		(void)device_remap(hptr_, dptr_, f);
		moved_from_obj_ = std::move(c);
	}

	/// return beginning of device_map
	iterator begin() const
	{
		return dptr_;
	}

	/// return end of device_map
	iterator end() const
	{
		return dptr_+size_;
	}

	/// return beginning of device_map as raw pointer
	T* data() const
	{
		return (T*)dptr_.get_base();
	}

	/// return size of device_map
	std::size_t size() const
	{
		return size_;
	}

	void debug()
	{
		std::cout << "dptr_: " << dptr_.get_base() <<
			" hptr_: " << hptr_ <<
			" size_: " << size_ <<
			std::endl;
	}

	device_ptr<T> dptr_;
	T* hptr_;
	std::size_t size_;
	boost::any moved_from_obj_;
	memory_tag tag_;
};

} // namespace aura
} // namespace boost

#endif // AURA_DEVICE_MAP_HPP

