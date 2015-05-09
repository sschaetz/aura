#include <memory>

namespace aura
{

template<typename BasePointer> 
struct pointer_rebind
{
	using type = BasePointer;
};
 
template<typename T> 
struct pointer_rebind<std::unique_ptr<T>>
{
	using type = std::unique_ptr<T, aura::backend::device_free<T>>;
};

template<typename T> 
struct pointer_rebind<std::shared_ptr<T>>
{
	using type = std::shared_ptr<T, aura::backend::device_free<T>>;
};


template<typename BasePointer> 
struct device_ptr_adaptor
{
	pointer_rebind<BasePointer>::type ptr_;
	// Put code from device_ptr here.
};
 
device_ptr_adaptor<int*> p = device_malloc(...);
device_free(p);
 
device_ptr_adaptor< unique_ptr<int> > p = make_unique(...);
device_ptr_adaptor< shared_ptr<int> > p = make_shared(...);
 
 
template<typename T>
using shared_device_ptr = device_ptr_adaptor< shared_ptr<T> >;
 
template<typename T>
using unique_device_ptr = device_ptr_adaptor< unique_ptr<T> >;
 
template<typename T>
using device_ptr = device_ptr_adaptor< T* >;
 
device_ptr<int> p = .... ;
shared_device_ptr<int> p = .... ; 

} // namespace aura

