#ifndef AURA_BACKEND_CUDA_MEMORY_HPP
#define AURA_BACKEND_CUDA_MEMORY_HPP

#include <cstddef>
#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/feed.hpp>
#include <aura/backend/cuda/device.hpp>
#include <aura/backend/cuda/device_ptr.hpp>


namespace aura {
namespace backend_detail {
namespace cuda {


/// memory handle
typedef CUdeviceptr memory;


/**
 * allocate device memory
 *
 * @param size the size of the memory
 * @param c the context the memory should be allocated in
 * @return a device pointer
 */
inline memory device_malloc(std::size_t size, device & d) {
  d.set();
  memory m;
  AURA_CUDA_SAFE_CALL(cuMemAlloc(&m, size));
  d.unset();
  return m;
}

DEPRECATED(memory device_malloc(std::size_t size, device & d));

template <typename T>
device_ptr<T> device_malloc(std::size_t size, device & d) {
  d.set();
  memory m;
  AURA_CUDA_SAFE_CALL(cuMemAlloc(&m, size*sizeof(T)));
  d.unset();
  return device_ptr<T>(m, d);
}


/**
 * free device memory
 *
 * @param m memory that should be freed
 */
inline void device_free(memory m, device & d) {
  d.set();
  AURA_CUDA_SAFE_CALL(cuMemFree(m));
  d.unset();
}

DEPRECATED(void device_free(memory m, device &));

template <typename T>
void device_free(device_ptr<T> & ptr) {
  ptr.get_device().set();
  AURA_CUDA_SAFE_CALL(cuMemFree(ptr.get()));
  ptr.get_device().unset();
  ptr.invalidate();
}


/**
 * copy host to device memory
 *
 * @param dst device memory (destination)
 * @param src host memory (source)
 * @param size size of copy in bytes
 * @param s stream the transfer is executed in
 * @param offset offset in bytes of the device memory
 */
inline void copy(memory dst, const void * src, std::size_t size, 
  feed & f, std::size_t offset=0) {
  f.set(); 
  AURA_CUDA_SAFE_CALL(cuMemcpyHtoDAsync(dst+offset, src, 
    size, f.get_backend_stream()));
  f.unset();
} 

DEPRECATED(void copy(memory dst, const void * src, std::size_t size, 
  feed & f, std::size_t offset));

/**
 * copy host to device memory
 *
 * @param dst device memory (destination)
 * @param src host memory (source)
 * @param size size of copy in number of T 
 * @param f feed the transfer is executed in
 */
template <typename T>
void copy(device_ptr<T> dst, const T * src, std::size_t size, 
  feed & f) {
  f.set(); 
  AURA_CUDA_SAFE_CALL(cuMemcpyHtoDAsync(
    dst.get()+dst.get_offset()*sizeof(T), src, 
    size*sizeof(T), f.get_backend_stream()));
  f.unset();
}


/**
 * copy device to host memory
 *
 * @param dst host memory (destination)
 * @param src device memory (source)
 * @param size size of copy in bytes
 * @param s stream the transfer is executed in
 * @param offset offset in bytes of the device memory
 */
inline void copy(void * dst, memory src, std::size_t size, 
  feed & f, std::size_t offset=0) {
  f.set();
  AURA_CUDA_SAFE_CALL(cuMemcpyDtoHAsync(dst, src+offset, 
    size, f.get_backend_stream()));
  f.unset();
}


DEPRECATED(inline void copy(void * dst, memory src, std::size_t size, 
  feed & f, std::size_t offset));

/**
 * copy device to host memory
 *
 * @param dst host memory (destination)
 * @param src device memory (source)
 * @param size size of copy in bytes
 * @param f feed the transfer is executed in
 */
template <typename T>
void copy(T * dst, const device_ptr<T> src, std::size_t size, feed & f) {
  f.set();
  AURA_CUDA_SAFE_CALL(cuMemcpyDtoHAsync(dst, 
    src.get()+src.get_offset()*sizeof(T), 
    size*sizeof(T), f.get_backend_stream()));
  f.unset();
}


/**
 * copy device to device memory
 *
 * @param dst host memory (destination)
 * @param src device memory (source)
 * @param size size of copy in bytes
 * @param s stream the transfer is executed in
 * @param offset offset in bytes of the device memory
 */
inline void copy(memory dst, memory src, std::size_t size, 
  feed & f, std::size_t dst_offset=0, std::size_t src_offset=0) {
  f.set();
  AURA_CUDA_SAFE_CALL(cuMemcpyDtoDAsync(dst+dst_offset, 
    src+src_offset, size, f.get_backend_stream()));
  f.unset();
}

DEPRECATED(void copy(memory dst, memory src, std::size_t size, 
  feed & f, std::size_t dst_offset, std::size_t src_offset));

/**
 * copy device to device memory
 *
 * @param dst host memory (destination)
 * @param src device memory (source)
 * @param size size of copy in bytes
 * @param f feed the transfer is executed in
 */
template <typename T>
inline void copy(device_ptr<T> dst, const device_ptr<T> src, 
  std::size_t size, feed & f) {
  f.set();
  AURA_CUDA_SAFE_CALL(cuMemcpyDtoDAsync(
    dst.get()+dst.get_offset()*sizeof(T), 
    src.get()+src.get_offset()*sizeof(T), size*sizeof(T), 
    f.get_backend_stream()));
  f.unset();
}


/**
 * Allocate memory on host for optimized host to device transfer
 *
 * @param size numter of bytes that should be allocated
 * @return pointer to allocated host memory
 */
void * host_malloc(const std::size_t& size)
{
	void * ptr;
	AURA_CUDA_SAFE_CALL(cuMemHostAlloc(&ptr,
		size, CU_MEMHOSTALLOC_PORTABLE));
	return ptr;
}

/**
 * Free memory on host allocated for optimized host to device transfer
 *
 * @param ptr the pointer that should be freed
 */
void host_free(void* ptr)
{
	AURA_CUDA_SAFE_CALL(cuMemFreeHost(ptr));
}

/**
 * Allocate memory on host for optimized host to device transfer
 *
 * @param num numter of T's that should be allocated
 * @return pointer to allocated host memory
 */
template <typename T>
T * host_malloc(const std::size_t& num)
{
	T * ptr;
	AURA_CUDA_SAFE_CALL(cuMemHostAlloc(reinterpret_cast<void**>(&ptr),
		num * sizeof(T), CU_MEMHOSTALLOC_PORTABLE));
	return ptr;
}

/**
 * Free memory on host allocated for optimized host to device transfer
 *
 * @param ptr the pointer that should be freed
 */
template <typename T>
inline void host_free(T* ptr)
{
	AURA_CUDA_SAFE_CALL(cuMemFreeHost(ptr));
}


} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_MEMORY_HPP

