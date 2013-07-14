#ifndef AURA_BACKEND_CUDA_MEMORY_HPP
#define AURA_BACKEND_CUDA_MEMORY_HPP

#include <cstddef>
#include <CL/cl.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/feed.hpp>
#include <aura/backend/cuda/device.hpp>


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
    size, f.get_stream()));
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
    size, f.get_stream()));
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
    src+src_offset, size, f.get_stream()));
  f.unset();
}


} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_MEMORY_HPP

