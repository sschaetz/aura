#include <tuple>
#include <iostream>

#include <boost/aura/bounds.hpp>
#include <boost/aura/slice.hpp>


using namespace boost::aura;

std::tuple<std::size_t, bounds> test(bounds b, slice idx)
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

void test2(bounds bin, slice iin)
{	
	bounds b;
	std::size_t offset;

	std::tie(offset, b) = test(bin, iin);
	std::cout << bin << " | " << iin << " || " << offset << " | " 
		<< b << std::endl;
}

int main(void) 
{
	test2(bounds(5,6,7), slice(_,3,4));
	test2(bounds(5,6,7), slice(_,_,4));
	test2(bounds(5,6,7), slice(_,_,_));
	test2(bounds(5,6,7), slice(2,3,4));
	test2(bounds(5,3,2), slice(_,2,1));
	test2(bounds(5,3,2,8), slice(_,2,2,0));
	test2(bounds(5,3,2,8), slice(_,2,2,1));
	test2(bounds(5,3,2,8), slice(0,0,0,0));
	test2(bounds(5,3,2,8), slice(0,0,0,1));
	test2(bounds(5,3,2,8), slice(1,0,0,0));
	test2(bounds(5,3,2,8), slice(_,0,0,0));
	test2(bounds(5,3,2,8), slice(_,_,0,0));
	test2(bounds(40,35,10), slice(_,1,1));
}
