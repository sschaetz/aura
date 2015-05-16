#include <memory>
#include <iostream>
#include <tuple>

#include <boost/aura/backend.hpp>

template <typename T>
using device_ptr = std::tuple<T*, std::ptrdiff_t, boost::aura::backend::device*>;


template <typename T>
device_ptr<T> operator +(const device_ptr<T>& a, const std::size_t inc)
{	
	auto tmp = a;
	std::get<1>(tmp) = std::get<1>(tmp) + inc;
	return tmp;
}


int main(void)
{
	device_ptr<float> p1;
	std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
	std::get<0>(p1) = &x[0];
	p1 = p1 + 2;

	std::cout << *(std::get<0>(p1) + std::get<1>(p1)) << std::endl;
	// std::cout << "hi!" << static_cast<std::size_t>(std::get<0>(p2)) << std::endl;
}

