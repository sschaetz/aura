#include <iostream>
#include <boost/any.hpp>

namespace aura {

template <typename T, typename Allocator>
T* addressof(std::vector<T, Allocator>& vec)
{
	return &vec[0];
}

template <typename T>
struct view;

template <typename T, typename C>
view<T> map(C& c);

template <typename T>
struct view {

	view() : ptr_(nullptr) {}
	~view() 
	{
		if (nullptr != ptr_) {
			// FIXME unmap the thing from the device
		}
	}
	template <typename C>
	view(C&c)
	{
		*this = map<T>(c);
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
	c = std::move(boost::any_cast<C>(v.moved_from_obj_));
	v.ptr_ = nullptr;
	// FIXME unmap the thing from the device
}

} // namespace aura

int main(void) 
{
	std::vector<float> h1(10, 1.);
	std::cout << &h1[0] << std::endl;
	aura::view<float> v1 = aura::map<float>(h1);
	std::cout << v1.ptr_ << " " << &h1[0] << std::endl;
	unmap(v1, h1);
	std::cout << v1.ptr_ << " " << &h1[0] << std::endl;
	
	std::vector<float> h2(10, 1.);
	std::cout << &h2[0] << std::endl;
	aura::view<float> v2(h2);
	std::cout << v2.ptr_ << " " << &h2[0] << std::endl;


}

