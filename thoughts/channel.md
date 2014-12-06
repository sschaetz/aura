Channel 
=======

* CUDA, seldom access, buffer provided, register host possible:
  register host, do DMA, no separate buffer on device
* CUDA, frequent access, buffer provided, register host possible:
  register host, do pinned memory copy, separate buffer on device

* CUDA, seldom access, buffer not provided, register host possible:
  allocate pinned host, do DMA, no separate buffer on device
* CUDA, frequent access, buffer not provided, register host possible:
  allocate pinned host, do pinned memory copy, separate buffer on device

* CUDA, seldom access, buffer provided, register host not possible:
  allocate pinned host, do copy to buffer then DMA, 
  no separate buffer on device
* CUDA, frequent access, buffer provided, register host not possible:
  allocate pinned host, do copyto buffer then pinned memory copy, 
  separate buffer on device

* CUDA, seldom access, buffer not provided, register host not possible:
  allocate pinned host, do DMA, no separate buffer on device
* CUDA, frequent access, buffer not provided, register host not possible:
  allocate pinned host, do pinned memory copy, separate buffer on device



device d(0);
feed f(d);
bounds b(100,100);

channel c(d, b);
std::vector<float> hv =<< f(c); // does this work?
wait_for(f);
// fill hv
device_array<float> dv << c(f) << hv;
wait_for(f);
// use dv


// -----

device d(0);
feed f(d);
bounds b(100,100);

channel c(d);
std::vector<float, allocator> hv = c.make_host<float>(b);
device_array<float> dv;

future<void> f1 = c.move(hv, dv);
future<void> f2 = invoke(kernel1, {512, 512}, 256, args(), f1);
future<void> f3 = invoke(kernel2, {512, 512}, 256, args(), f1);


