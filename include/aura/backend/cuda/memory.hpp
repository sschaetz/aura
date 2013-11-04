#ifndef AURA_BACKEND_CUDA_MEMORY_HPP
#define AURA_BACKEND_CUDA_MEMORY_HPP

#include <cstddef>
#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/feed.hpp>
#include <aura/backend/cuda/device.hpp>


namespace aura {
namespace backend_detail {
namespace cuda {


/// memory handle
typedef CUdeviceptr memory;

#if 0
typedef CUdevicememory memory;
// FIXME: let's assume we can do pointer arithmetic with CUdevicememory_

template <typename T>
struct device_memory_ {

  /// pointer type
  typedef T * pointer;

  /// const pointer type
  typedef const T * const_pointer;

  /// value type the device pointer holds
  typedef T value_type;

public:

 /**
  * @brief create pointer that points nowhere
  */
  device_memory_() : device_(nullptr) {}

 /**
   * @brief create device pointer that points to memory_
   *
   * @param m memory that identifies device memory
   * @param d device the memory is allocated on 
   */
  device_memory_(memory & m, device & d) : memory_(m), 
    device_(&d) {}

  /**
   * @brief returns a pointer to the device memory 
   */
  pointer get() { return reinterpret_cast<pointer>(memory_); }
  
  /**
   * @brief returns a pointer to the device memory 
   */
  const_pointer get() { return reinterpret_cast<const_pointer>(memory_); }


  /**
   * @brief returns the memory 
   */
  memory get_memory() { return memory_ }
  
  /**
   * @brief returns the memory 
   */
  const memory get_memory() const { return memory_ }

  /**
   * @brief assign other device_ptr to this device_ptr
   *
   * @param b device_memory_ that should be assigned
   * @return reference to this device_memory_
   */
  device_memory_<T>& operator =(device_ptr<T> const & b)
  {
    memory_ = b.memory_;
    device_ = b.device_;
    return *this;
  }

  /**
   * @brief increment the device pointer and return a new one
   *
   * @param b value by which the pointer should be incremented
   * @return incremented device pointer
   */
  device_memory_<T> operator +(const std::size_t & b) const
  {
    memory x = memory_ + b;
    return device_ptr<T>(x, device_);
  }

  /**
   * @brief increment the device pointer and return it
   *
   * @param b value by which the pointer should be incremented
   * @return incremented device pointer
   */
  device_memory_<T>& operator +=(const std::size_t & b)
  {
    memory_+=b;
    return *this;
  }

  /**
   * @brief increment the device pointer by one
   *
   * @return incremented device pointer
   */
  device_memory_<T> & operator ++()
  {
    ++memory_;
    return *this;
  }

  /**
   * @brief decrement the device pointer by one
   *
   * @return decrement device pointer
   */
  device_memory_<T> & operator --()
  {
    --memory_;
    return *this;
  }

  /**
   * @brief decrement the device pointer and return a new one
   *
   * @param b value by which the pointer should be decremented
   * @return decrement device pointer
   */
  device_memory_<T> operator -(const std::size_t & b) const
  {
    memory x = memory_ - b;
    return device_ptr<T>(x, device_);
  }

  /**
   * @brief subtract one memory_ from another memory_
   *
   * @param other pointer
   * @return difference in elements between pointers
   */
  std::memory_diff_t operator -(const device_memory_<T> & b) const
  {
    assert(device_ != b.devce_) 
    return reinterpret_cast<pointer>(memory_) - 
      reinterpret_cast<pointer>(b.memory_);
  }

  /**
   * @brief compare two device pointers
   *
   * @param b pointer that should be compared
   * @return true if pointers are equal, else false
   */
  bool operator ==(const device_memory_<T> & b) const
  {
    return (device_ == b.device_ && memory_ == b.memory_);
  }

  /**
   * @brief compare two device pointers
   *
   * @param b pointer that should be compared
   * @return true if pointers are unequal, else false
   */
  bool operator !=(const device_memory_<T> & b) const
  {
    return !(*this == b);
  }

private:

  /// actual pointer that identifies device memory
  memory memory_;

  /// reference to device the feed was created for
  device * device_;

};
#endif


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
    size, f.get_backend_stream()));
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


} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_MEMORY_HPP

