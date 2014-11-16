#include <iostream>
#include <boost/any.hpp>

// Experiment: see if moveable and any work well together
// see if we can move the gut out of an instance when mapping
// it and putting it back in when unmapping
//
// Result: seems to work. any_cast must be passed and rvalue.

namespace boost 
{
namespace aura 
{

template <typename T, typename Allocator>
T* addressof(std::vector<T, Allocator>& vec)
{
	return &vec[0];
}

template <typename T>
T* addressof(T* ptr)
{
	return ptr;
}

template <typename T>
struct view;

template <typename T, typename C>
view<T> map(C& c);

template <typename T>
struct view {
	// fixme: should be movable but not copyable
	view() : ptr_(nullptr) {}

	view(view&& v) : 
		ptr_(v.ptr_),
		moved_from_obj_(std::move(v.moved_from_obj_))
	{
		v.ptr_ = nullptr;
	}

	view<T>& operator= (const view<T>& other)
	{
		ptr_ = other.ptr_;
		moved_from_obj_ = std::move(other.moved_from_obj_);
		other.ptr_ = nullptr;
	}

	template <typename C> 
	view(C& c)
	{
		ptr_ = addressof(c);
		moved_from_obj_ = std::move(c);
	}

	~view() 
	{
		if (nullptr != ptr_) {
			// FIXME unmap the thing from the device
		}
	}

	T* ptr_;
	boost::any moved_from_obj_;
};

template <typename T, typename C>
view<T> map(C& c)
{
	view<T> r;
	r.ptr_ = addressof(c);
	r.moved_from_obj_ = std::move(c);
	return r;
}

template <typename T, typename C>
void unmap(view<T>& v, C& c)
{
	c = boost::any_cast<C>(std::move(v.moved_from_obj_));
	v.ptr_ = nullptr;
	// FIXME unmap the thing from the device
}

} // namespace aura
} // namespace boost 

int main(void) 
{
	{
		std::vector<float> h1(10, 1.);
		std::cout << &h1[0] << std::endl;
		boost::aura::view<float> v1 = boost::aura::map<float>(h1);
		std::cout << v1.ptr_ << " " << &h1[0] << std::endl;
		unmap(v1, h1);
		std::cout << v1.ptr_ << " " << &h1[0] << std::endl;
	}
	
	{
		std::vector<float> h2(10, 1.);
		std::cout << &h2[0] << std::endl;
		{
			boost::aura::view<float> v2(h2);
			std::cout << v2.ptr_ << " " << &h2[0] << std::endl;
		}
		std::cout << &h2[0] << std::endl;
	}

	{
		float h[100];
		boost::aura::view<float> v1 = boost::aura::map<float>(h);
		std::cout << v1.ptr_ << " " << h << std::endl;
	}
}

