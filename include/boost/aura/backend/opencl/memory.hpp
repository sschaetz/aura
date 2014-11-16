#ifndef AURA_BACKEND_OPENCL_MEMORY_HPP
#define AURA_BACKEND_OPENCL_MEMORY_HPP

#include <CL/cl.h>
#include <boost/aura/backend/opencl/call.hpp>
#include <boost/aura/backend/opencl/feed.hpp>
#include <boost/aura/backend/opencl/device.hpp>
#include <boost/aura/backend/opencl/device_ptr.hpp>
#include <boost/aura/backend/shared/memory_tag.hpp>
#include <boost/aura/misc/deprecate.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace opencl {


/// memory handle
typedef cl_mem memory;

/**
 * translates an Aura memory tag to an OpenCL memory tag
 */
inline cl_mem_flags translate_memory_tag(memory_tag tag) 
{
	cl_mem_flags flag = CL_MEM_READ_WRITE;
	if (tag == memory_tag::ro) {
		flag = CL_MEM_READ_ONLY;
	} else if (tag == memory_tag::ro) {
		flag = CL_MEM_WRITE_ONLY;
	}
	return flag;
}

inline cl_map_flags translate_map_tag_inverted(memory_tag tag) 
{
	cl_mem_flags flag = CL_MAP_READ | CL_MAP_WRITE;
	if (tag == memory_tag::ro) {
		flag = CL_MAP_WRITE;
	} else if (tag == memory_tag::ro) {
		flag = CL_MAP_READ;
	}
	return flag;
}
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
 * map host memory to device memory 
 *
 * @param ptr pointer to host memory
 * @param size number of T's that should be mapped
 * @param tag memory tag of the map 
 * @param d device the memory should be mapped to
 *
 * @return device pointer corresponding to the mapped region
 */
template <typename T>
device_ptr<T> device_map_alloc(T* ptr, std::size_t size, 
		memory_tag tag, device& d)
{
	int errorcode = 0;
	cl_mem_flags flag = translate_memory_tag(tag);
	typename device_ptr<T>::backend_type m =
		clCreateBuffer(d.get_backend_context(),
		flag | CL_MEM_USE_HOST_PTR,
		size*sizeof(T), ptr, &errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	return device_ptr<T>(m, d);
}

/**
 * remap previsouly created and unmapped map back to device memory
 *
 * @param ptr pointer to host memory
 * @param dptr pointer to device memory
 * @param size number of T's that should be mapped
 * @param tag memory tag of the map 
 * @param d device the memory should be mapped to
 *
 * @return device pointer corresponding to the mapped region
 */
template <typename T>
device_ptr<T> device_remap(T* ptr, device_ptr<T> dptr, feed& f)
{
	AURA_OPENCL_SAFE_CALL(clEnqueueUnmapMemObject(f.get_backend_stream(), 
				dptr.get(), ptr, 0, NULL, NULL));
	return dptr;	
}

/**
 * unmap memory that was previously mapped to to a device 
 *
 * @param ptr pointer to host memory that should be unmapped
 * @param dptr device pointer corresponding to mapped region 
 * @param size size of memory region that should be unmapped
 * @param tag memory tag of the map 
 * @param f feed that should be used for the data transfer
 */
template <typename T>
void device_unmap(T* ptr, device_ptr<T>& dptr, 
		std::size_t size, feed& f) 
{
	int errorcode = 0;
	// if the memory was mapped to the device for reading,
	// nothing needs to be done
	if (dptr.get_memory_tag() != memory_tag::ro) {
		void* r = clEnqueueMapBuffer(f.get_backend_stream(), 
				dptr.get(), CL_FALSE, 
				CL_MAP_READ|CL_MAP_WRITE, 
				dptr.get_offset(),
				size*sizeof(T),
				0, NULL, NULL, &errorcode);
		// if this does not hold, we are in deep trouble
		assert(r == (void*)ptr);
		AURA_OPENCL_CHECK_ERROR(errorcode);
	}
	return;
}

template <typename T>
void device_map_free(T* ptr, device_ptr<T>& dptr)
{
	device_free(dptr);
	dptr = nullptr;
	return;
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


/**
 * Allocate memory on host for optimized host to device transfer
 *
 * @param size numter of bytes that should be allocated
 * @return pointer to allocated host memory
 */
inline void * host_malloc(const std::size_t& size)
{
	return malloc(size);
}

/**
 * Free memory on host allocated for optimized host to device transfer
 *
 * @param ptr the pointer that should be freed
 */
inline void host_free(void* ptr)
{
	free(ptr);
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
	return (T*)malloc(sizeof(T) * num);
}

/**
 * Free memory on host allocated for optimized host to device transfer
 *
 * @param ptr the pointer that should be freed
 */
template <typename T>
inline void host_free(T* ptr)
{
	free((void*)ptr);
}


} // opencl 
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_OPENCL_MEMORY_HPP

