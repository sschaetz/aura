# Memory Access Functions

Function names to access memory are a mess currently in Aura. My initial
thoughts on this are appended to this documents [0].

Problem 1:
Instead of 
T -> T* -> iterator -> container
Aura has
T -> T* -> device_ptr -> container
with device_ptr being a custom pointer instead of an actual iterator.

Problem 2:
Pointer arithmetic is not trivial in OpenCL. OpenCL memory pointers
are no real pointers but only handles. Arithmetic on them just does not work.
Thus device_ptr stores a base pointer (or the handle) and an offset.


Here is a proposed solution:

device_ptr provides
get_base(), get_offset(), get_device(), get_memory_tag()
functions, get() is thus deprecated (replaced by get_base).

device_array provides
begin() -> device_ptr
end() -> device_ptr
data() -> T*

device_array<foo> x;
x.begin().get_base() 
is thus the same as 
x.data()
and well formed.

The following however is an error:

auto it = x.begin();
std:;advance(it, 5);
it.get_base()
is still equal to x.begin()
to get the correct address, the offset must be considered:
it.get_base() + it.get_offset()




[0]
Names are clearly not sensible yet in Aura. Here are rules for naming
member functions for common tasks:

There is the question of having get_ style functions or without get
functions.

So I'm thinking get_* for Aura types and get_backend_* for accessing
backend types. This is already implemented.

The STL has

vector<T>:
	get_allocator (that speaks for get_ functions)
	data() -> T*
	begin(), end() -> iterator
	operator[] -> T
	front(), back() -> T

Now Aura has the device_ptr<T> in-between the container and T*.
The questions are:
	should there be a proper iterator or should device_ptr simply
	implement the iterator interface, possibly using the boost
	iterator helpers?

Maybe having a data() member to access the raw handle is a good
idea? Let's look at shared_ptr - it has the get() member.
To pass a raw-handle to a kernel one would have to call
A.begin().get() which is ugly
A.data() is cooler
but seems inconsistent.
Invoke could of course resolve the iterator to the raw handle but then
the kernel interface and the invoke call do not match which is confusing.
Then again, only on the iterator type is arithmetic possible, the OpenCL
backend would not allow A.data() + 15, instead it should be
std::advance(A.begin(), 15)




