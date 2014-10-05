# Zero-copy

There are 3 different things to consider:

1) no zero-copy is desired but and actual copy
	-> use copy()
2) zero-copy is desired and available:
	-> use move() (or something similar)
3) zero-copy is desired but not available:
	-> a move() maps a host-pointer device accessible
	   but it is not clear if an actual copy is desired

Alternative:
A view could move out all the data from a host container, 
store it while it is mapped, and put it back when it is unmapped
or destroyed.
This however does not work with raw pointers or 
const raw pointers - or maybe it does, they are just not invalidated.

so:

T foo[100];
{
	device_view<T> v = map(foo, 100); 
	// or
	device_view<T> v(foo, 100);
	// don't use foo, use v
	v.unmap();
	// use foo
	v.map();
	// use v
}
// use foo

if v is accessed from within a kernel
if the device supports zero-copy all is well
if the device doesn't, it accesses the memory over the bus

but that is maybe not desired, because in case of no zero-copy
an actual copy might be faster.

What to do?
The view could have a policy saying: if possible do zero-copy,
if not, create an actual copy, and: do zero-copy (mapping) in
any case.

Maybe this constructor:
template <typename T>
device_view(T* ptr, std::size_t size, int behaviour = opportunistic)
{}

with possible values for behaviours:
opportunistic (zero-copy if possible, otherwise copy)
zerocopy (always do zero-copy, accelerator has to go through 
	bus if zero-copy is not available)



