
#include <vector>
#include <stdio.h>

// test how overloading of a typecast works

// nice, this seems to work
// so we can provide an std::vector with a bounds() type

// what happens if there is a ctor for the original type and 
// the type that can be casted to?

// nice, this also seems to work
// it call the ctor without the typecast

struct foo
{
	foo(int x, int y, int z) : x_(x), y_(y), z_(z) 
	{}

	operator std::size_t() {return (std::size_t)(x_*y_*z_);}

private:
	int x_;
	int y_;
	int z_;
};


struct bar
{
	explicit bar(std::size_t size)
	{
		printf("size ctor called\n");
	}
	
	// if this ctor is removed, it calls the size ctor twice
	explicit bar(foo f)
	{
		printf("foo ctor called\n");
	}
	
};

int main(void) {
	std::vector<float> vec(foo(1,2,3));
	printf("%lu\n", vec.size());

	bar(100);
	bar(foo(1,2,3));
}

