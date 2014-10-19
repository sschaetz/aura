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


## API

CUDA:
CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags)
CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, 
	unsigned int Flags)
CUresult cuMemHostUnregister(void* p)

OpenCL:
clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, 
	void* host_ptr, cl_int* errcode_ret)
cl_int clReleaseMemObject(cl_mem memobj)

## Differences between CUDA and OpenCL

CUDA seems to transfer the data back and forth within the kernel call.
OpenCL seems to buffer the kernel writing part so it has to be 
explicitly "mapped" back to host memory. For this a feed must be specified.


## Problems:

The user should be able to map a region of memory but never unmap it. That
is, the user wants to provide memory to the device in a read only fashion.
However, if it is implemented with a move, the host object never gets its 
content back. So another function might be necessary that says: ok, don't 
transfer the data back to the host, but make sure nobody reads the view any 
longer and give me my block of memory back.

The map/unmap pair of functions resembles malloc/free. This is not good since
we wanted to get rid of these things. Forgetting to call free/unmap is a real
problem.

Maybe we need a concept that does the free automatically when it dies. Like
a range/iterator. Hmhm.

what about
device_view<T, ro>(vec, d); // does not sync back
device_view<T, rw>(vec, d); // does sync back (default)
device_view<T, wo>(vec, d); // does not sync forward (does not exist) 

in all three cases, vec is moved from and in the dtor, is moved back into.
But that means we need a
device_view<std::vector<T>> 
so we can get rid of the any. Or do we?

Then we could always call std::begin()
and for raw pointers we could offer a

std::make_tupe(T* begin, std::size_t size) or
std::make_tupe(T* begin, T* end)

and
template <typename T, typename C>
T* startaddress(C& obj)
would by default call
std::begin(obj) and std::distance(std::begin(obj), std::end(obj))

but an overload should be provided for the tuple case

The value type we can get through
iterator_traits<T>::value_type
even for the pointer case

