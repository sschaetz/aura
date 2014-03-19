# Aura

Copyright (C) 2011-2013 Biomedizinische NMR Forschungs GmbH


Aura is a modern, header-only C++ library for accelerator development. Aura 
works with both OpenCL and CUDA backends. The Aura API is not stable yet (alpha
version).

## Concepts

Aura has only a few core concepts:

The `device` type is the starting point for all interactions with accelerators. It represents the device context known from OpenCL and CUDA. 

There are two layers of abstraction for memory management. A typed pointer to device memory `device_ptr<T>` is the low level memory abstraction. The type contains a raw pointer that represents a location in accelerator memory and a reference to a `device`. To enable pointer arithmetic, the OpenCL implementation of `device_ptr<T>` also contains an offset from the base pointer.

~~~{.cpp}
device d(0);
device_ptr<int> ptr = device_malloc<int>(16, d);
device_free(ptr);
~~~

The second memory management layer provides containers. A `device_array<T>` extends the `device_ptr<T>` type with the `bounds` type. `bounds` defines a multidimensional discrete space. A `device_array<T>` represents a continuous block of device memory of size $\prod_{j=0}^{J-1}$`bounds[j]}.

~~~{.cpp}
device d(0);
device_array<int> array1(40, d);
device_array<int> array2(bounds(40, 20, 10), d);
~~~

Commands such as memory transfers and kernel invocations are issued to an accelerator through `feeds`. A `feed` is a queue of commands that specifies both location and order of execution of commands. A `feed` is always associated with a single `device`. There can exists multiple `feeds` for each `device`. Commands issued to different `feeds` of the same `device` execute concurrently and in the same memory address space. If supported by the accelerator, concurrent commands can execute in parallel. The `wait_for` function and `wait` method block the caller until all commands issued to a feed have finished.

~~~{.cpp}
std::vector<float> src(product(bounds(40, 20)));
device d(0);
device_array<int> dst(bounds(40, 20), d);
feed f(d);
copy(dst, src, f);
f.wait(); /* blocking until copy finished */
wait_for(f); /* alternative to f.wait(); */
~~~

The `copy` function is the only function required to transfer data between host and accelerator and between accelerators. Since the compiler can discriminate between accelerator and host memory, the correct `copy` function is dispatched at compile time. Both an iterator/pointer based and a range based interface are supported.

~~~{.cpp}
copy(dst.begin(), dst.end(), src.begin(), f);
copy(dst, src, f);
~~~

The `invoke` function parameterizes and enqueues kernels in a feed. Its first argument is a `kernel` object created from a `module`. The second argument specifies the number of `fibers` that should be launched using a `mesh`. The third argument partitions the `mesh` in `bundles` of `fibers`. The fourth argument is a tuple
containing arguments that should be passed to the kernel. The last argument specifies the `feed` the kernel should be enqueued in.

~~~{.cpp}
int xdim = 128; int dimy = 64;
device d(0);
module m = create_module_from_file("k.cl", d);
kernel k = create_kernel(m, "simple_add");
device_array<int> v(bounds(xdim, dimy), d);
feed f(d);
invoke(k, mesh(dimy, dimx), bundle(dimx), args(vec.begin_raw()), f); 
wait_for(f);
~~~

If the space of `fibers` can be partitioned arbitrarily, that is, if the kernel contains no synchronization points, kernel invocation can be simplified. The `invoke` function can determine how to best partition the `fiber`-space based on platform properties and heuristics.

~~~{.cpp}
invoke(k, bounds(dimx, dimy), args(v.begin_raw()), f);
/* or: */
invoke(k, v.get_bounds(), 
    args(v.begin_raw()), f);
~~~

A `mark` allows orchestrating and synchronizing multiple `feeds`. A `mark` is inserted into a `feed`. It can either be waited for from the calling thread or another `feed` can be instructed to wait for a mark through its `continue_when` member.

~~~{.cpp}
device d(0);  
feed f1(d);
feed f2(d);
mark m1;
insert(f1, m1);
/* or: */
mark m2(f2);
wait_for(m1);
f1.continue_when(m2);
~~~
now event wait_for(e) f.wait_for(e) wait_for(f, e) (e must be enqueued in f, otherwise this does not work in OpenCL).


----------------------------

Author: Sebastian Schaetz <seb.schaetz@gmail.com>

Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE.md or copy at 
[boost.org/LICENSE_1_0.txt](http://www.boost.org/LICENSE_1_0.txt "boost.org/LICENSE_1_0.txt"))



