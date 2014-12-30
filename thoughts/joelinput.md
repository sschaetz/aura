//RAII
device d(0);
device_ptr<int> ptr = device_malloc<int>(16, d);
 
template<typename BasePointer> struct pointer_rebind
{
using type = BasePointer
};
 
template<typename T> struct pointer_rebind< unique_ptr<T> >
{
using type = unique_ptr<T,my_device_deleter>;
};
 
template<typename BasePointer> struct device_ptr_adaptor
{
pointer_rebind<BasePointer>::type ptr_;
-> returns ptr_;
* retrns *ptr_;
++
+=
+
// probably need swap and move for suporting unique_ptr
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




device d(0);
int xdim = 128; int dimy = 64;
 
device_array<int> v(bounds(xdim, dimy), d);
 
using module = std::unordered_map<kernel,string>;
module m;
 
m.insert( d.load("simple_add","k.cl") );
 
// the abstraction shoudl be fire kernels/commands in an asynchronous context
// async handles asynhronous
// d tells you where to shove async
 
feed f = async( m["simple_add"], d, {dimy, dimx}, bundle(dimx), vec );
feed g = async( your copy stuff , d, {dimy, dimx}, bundle(dimx), vec );
 
// that"s teh inteface from C++17 future
f.then(g).then(h);
 
// OR you use promises
/ promises encapsulates futures
// in a way that then is automatic
 
feed f = async( m["simple_add"], d, {dimy, dimx}, bundle(dimx), vec );
feed g = async( your copy stuff , d, {dimy, dimx}, bundle(dimx), f, f ); 



