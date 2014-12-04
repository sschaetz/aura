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



