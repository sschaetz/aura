#ifndef AURA_BACKEND_CUDA_MEMORY_HPP
#define AURA_BACKEND_CUDA_MEMORY_HPP

#include <CL/cl.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/context.hpp>
#include <aura/backend/cuda/stream.hpp>


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
inline memory device_malloc(std::size_t size, context c) {
  AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(c));
  memory m;
  AURA_CUDA_SAFE_CALL(cuMemAlloc(&m, size));
  AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(NULL));
  return m;
}


/**
 * free device memory
 *
 * @param m memory that should be freed
 */
inline void device_free(memory m) {
  AURA_CUDA_SAFE_CALL(cuMemFree(m));
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
  stream s, std::size_t offset=0) {
  AURA_CUDA_SAFE_CALL(cuMemcpyHtoDAsync(dst+offset, src, size, s));
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
  stream s, std::size_t offset=0) {
  AURA_CUDA_SAFE_CALL(cuMemcpyDtoHAsync(dst, src+offset, size, s));
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
  stream s, std::size_t dst_offset=0, std::size_t src_offset=0) {
  AURA_CUDA_SAFE_CALL(cuMemcpyDtoDAsync(dst+dst_offset, 
    src+src_offset, size, s));
}


} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_MEMORY_HPP

