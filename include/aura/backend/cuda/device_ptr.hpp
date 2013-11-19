#ifndef AURA_BACKEND_CUDA_DEVICE_PTR_HPP
#define AURA_BACKEND_CUDA_DEVICE_PTR_HPP

#include <cstddef>
#include <cuda.h>
#include <aura/backend/cuda/device.hpp>
#include <aura/backend/cuda/memory.hpp>


namespace aura {
namespace backend_detail {
namespace cuda {


template <typename T>
struct device_ptr {

  /// backend handle type
  typedef CUdeviceptr backend_type;
  
  /// backend handle type
  typedef const CUdeviceptr const_backend_type;

public:

 /**
  * @brief create pointer that points nowhere
  */
  device_ptr() : memory_(nullptr), device_(nullptr) {}

 /**
  * @brief create device pointer that points to memory_
  *
  * @param m memory that identifies device memory
  * @param d device the memory is allocated on 
  */
  device_ptr(backend_type & m, const device & d) : 
    memory_(m), device_(const_cast<device *>(&d)) {}

  /// returns a pointer to the device memory 
  backend_type get() { return memory_; }
  
  /// returns a pointer to the device memory 
  const_backend_type get() const { return memory_; }

  /// assign operator
  device_ptr<T>& operator =(device_ptr<T> const & b) {
    memory_ = b.memory_;
    device_ = b.device_;
    return *this;
  }

  /// addition operator
  device_ptr<T> operator +(const std::size_t & b) const {
    memory x = memory_ + b;
    return device_ptr<T>(x, *device_);
  }

  /// addition assignment operator
  device_ptr<T>& operator +=(const std::size_t & b) {
    memory_+=b;
    return *this;
  }

 	/// prefix addition operator
  device_ptr<T> & operator ++() {
    ++memory_;
    return *this;
  }

  /// postfix addition operator
  device_ptr<T> operator ++(int) {
    memory x = memory_ + 1;
    return device_ptr<T>(x, *device_);
  }
 
  /// subtraction operator
  device_ptr<T> operator -(const std::size_t & b) const {
    memory x = memory_ - b;
    return device_ptr<T>(x, *device_);
  }

  /// subtraction assignment operator
  device_ptr<T>& operator -=(const std::size_t & b) {
    memory_-=b;
    return *this;
  }

 	/// prefix subtraction operator
  device_ptr<T> & operator --() {
    --memory_;
    return *this;
  }

  /// postfix subtraction operator
  device_ptr<T> operator --(int) {
    return device_ptr<T>(memory_-1, *device_);
  }

  /// equal to operator
  bool operator ==(const device_ptr<T> & b) const  {
    return (device_->get_ordinal() == b.device_->get_ordinal() && 
      memory_ == b.memory_);
  }

  /// not equal to operator
  bool operator !=(const device_ptr<T> & b) const {
    return !(*this == b);
  }

private:

  /// actual pointer that identifies device memory
  memory memory_;

  /// reference to device the feed was created for
  device * device_;

};

} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_DEVICE_PTR_HPP

