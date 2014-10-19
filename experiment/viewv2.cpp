
#include <tuple>
#include <vector>
#include <iostream>

// Experiment: see how a container and range based interface
// can be combined

template <typename C>
std::ptrdiff_t get_diff(const C& c)
{
	return std::distance(c.begin(), c.end());
}

template <typename T>
std::ptrdiff_t get_diff(const std::tuple<T*, T*>& c)
{
	return std::distance(std::get<0>(c), std::get<1>(c));
}

template <typename T>
std::ptrdiff_t get_diff(const std::tuple<T*, std::size_t>& c)
{
	return std::get<1>(c);
}

template <typename T>
std::ptrdiff_t get_diff(const std::tuple<T*, int>& c)
{
	return std::get<1>(c);
}

template <typename T>
struct view
{
	template <typename C>
	view(C& c) : diff_(get_diff(c))
	{
		std::cout << "ctor " << diff_ << std::endl;
	}

	template <typename C>
	view(const C&& c) : diff_(get_diff(c))
	{
		std::cout << "ctor " << diff_ << std::endl;
	}
	std::ptrdiff_t diff_;
};

int main(void) 
{
	std::vector<float> vec1(10, 4.2);
	view<float> v1(vec1);

	view<float> v2(std::make_tuple(&vec1[0], &vec1[vec1.size()]));

	view<float> v3(std::make_tuple(&vec1[0], 4));

}

