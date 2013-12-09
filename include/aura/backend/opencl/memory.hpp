#ifndef AURA_BACKEND_OPENCL_MEMORY_HPP
#define AURA_BACKEND_OPENCL_MEMORY_HPP

#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/feed.hpp>
#include <aura/backend/opencl/device.hpp>
#include <aura/backend/opencl/device_ptr.hpp>
#include <aura/misc/deprecate.hpp>


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
inline memory device_malloc(std::size_t size, device & d) {
  int errorcode = 0;
  memory m = clCreateBuffer(d.get_backend_context(), CL_MEM_READ_WRITE, 
    size, 0, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  return m;
}

DEPRECATED(memory device_malloc(std::size_t size, device & d));

template <typename T>
device_ptr<T> device_malloc(std::size_t size, device & d) {
  int errorcode = 0;
  typename device_ptr<T>::backend_type m = 
    clCreateBuffer(d.get_backend_context(), 
    CL_MEM_READ_WRITE, size*sizeof(T), 0, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  return device_ptr<T>(m, d);
}

/**
 * free device memory
 *
 * @param m memory that should be freed
 */
inline void device_free(memory m, device &) {
  AURA_OPENCL_SAFE_CALL(clReleaseMemObject(m));
}

DEPRECATED(void device_free(memory m, device &));

template <typename T>
void device_free(device_ptr<T> & ptr) {
  AURA_OPENCL_SAFE_CALL(clReleaseMemObject(ptr.get()));
  ptr.invalidate();
}

/**
 * copy host to device memory
 *
 * @param dst device memory (destination)
 * @param src host memory (source)
 * @param size size of copy in bytes
 * @param f feed the transfer is executed in
 * @param offset offset in bytes of the device memory
 */
inline void copy(memory dst, const void * src, std::size_t size, 
  feed & f, std::size_t offset=0) {
  AURA_OPENCL_SAFE_CALL(clEnqueueWriteBuffer(f.get_backend_stream(),
  	dst, CL_FALSE, offset, size, src, 0, NULL, NULL));
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
  AURA_OPENCL_SAFE_CALL(clEnqueueWriteBuffer(f.get_backend_stream(),
  	dst.get(), CL_FALSE, dst.get_offset()*sizeof(T), size*sizeof(T), 
    src, 0, NULL, NULL));
}

/**
 * copy device to host memory
 *
 * @param dst host memory (destination)
 * @param src device memory (source)
 * @param size size of copy in bytes
 * @param f feed the transfer is executed in
 * @param offset offset in bytes of the device memory
 */
inline void copy(void * dst, memory src, std::size_t size, 
  feed & f, std::size_t offset=0) {
  AURA_OPENCL_SAFE_CALL(clEnqueueReadBuffer(f.get_backend_stream(),
  	src, CL_FALSE, offset, size, dst, 0, NULL, NULL));
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
  AURA_OPENCL_SAFE_CALL(clEnqueueReadBuffer(f.get_backend_stream(),
  	src.get(), CL_FALSE, src.get_offset()*sizeof(T), size*sizeof(T), 
    dst, 0, NULL, NULL));
}

/**
 * copy device to device memory
 *
 * @param dst host memory (destination)
 * @param src device memory (source)
 * @param size size of copy in bytes
 * @param f feed the transfer is executed in
 * @param offset offset in bytes of the device memory
 */
inline void copy(memory dst, memory src, std::size_t size, 
  feed & f, std::size_t dst_offset=0, std::size_t src_offset=0) {
  AURA_OPENCL_SAFE_CALL(clEnqueueCopyBuffer(f.get_backend_stream(),
    src, dst, src_offset, dst_offset, size, 0, 0, 0)); 	
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
  AURA_OPENCL_SAFE_CALL(clEnqueueCopyBuffer(f.get_backend_stream(),
    src.get(), dst.get(), src.get_offset()*sizeof(T), 
    dst.get_offset()*sizeof(T), size*sizeof(T), 
    0, 0, 0)); 	
}

} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_MEMORY_HPP

