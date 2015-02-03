#include <tuple>
#include <iostream>

#include <boost/aura/bounds.hpp>
#include <boost/aura/index.hpp>


using namespace boost::aura;

std::tuple<std::size_t, bounds> test(bounds b, index idx)
{
	bounds r;
	int cutoff = -1;
	for (std::size_t i=0; i<idx.size(); i++) {
		if (idx[i] == _) {
			r.push_back(b[i]);
		} else if (cutoff == -1) {
			cutoff = i;	
		}
	}
	std::size_t offset = 0;
	std::cout << cutoff << std::endl;
	// multiply base dimensions with index of offset
	if (cutoff >= 0) {
		offset = product(b.take(cutoff));
		offset *= product(idx.drop(idx.size()-cutoff));
	// one element specified, offset is 
	} else {
		offset = product(b)
	}
	return std::make_tuple(offset, r);
}

void test2(bounds bin, index iin)
{	
	bounds b;
	std::size_t offset;

	std::tie(offset, b) = test(bin, iin);
	std::cout << bin << " | " << iin << " || " << offset << " | " 
		<< b << std::endl;
}

int main(void) 
{
	test2(bounds(5,6,7), index(_,3,4));
	test2(bounds(5,6,7), index(_,_,4));
	test2(bounds(5,6,7), index(_,_,_));
	test2(bounds(5,6,7), index(2,3,4));
}
