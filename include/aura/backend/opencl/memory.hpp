#ifndef AURA_BACKEND_OPENCL_MEMORY_HPP
#define AURA_BACKEND_OPENCL_MEMORY_HPP

#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/feed.hpp>


namespace aura {
namespace backend_detail {
namespace opencl {


/// memory handle
typedef cl_mem memory;


/**
 * allocate device memory
 *
 * @param size the size of the memory
 * @param c the context the memory should be allocated in
 * @return a device pointer
 */
inline memory device_malloc(std::size_t size, feed & f) {
  int errorcode = 0;
  memory m = clCreateBuffer(f.get_context(), CL_MEM_READ_WRITE, 
    size, 0, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  return m;
}


/**
 * free device memory
 *
 * @param m memory that should be freed
 */
inline void device_free(memory m, feed &) {
  AURA_OPENCL_SAFE_CALL(clReleaseMemObject(m));
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
  AURA_OPENCL_SAFE_CALL(clEnqueueWriteBuffer(f.get_stream(),
  	dst, CL_FALSE, offset, size, src, 0, NULL, NULL))
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
  AURA_OPENCL_SAFE_CALL(clEnqueueReadBuffer(f.get_stream(),
  	src, CL_FALSE, offset, size, dst, 0, NULL, NULL));
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
  AURA_OPENCL_SAFE_CALL(clEnqueueCopyBuffer(f.get_stream(),
    src, dst, src_offset, dst_offset, size, 0, 0, 0)); 	
}


} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_MEMORY_HPP

